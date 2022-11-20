use std::collections::HashMap;

use rush_analyzer::{ast::*, InfixOp, Type};

use crate::{
    instruction::{Instruction, Section},
    register::{FloatRegister, IntRegister, FLOAT_PARAM_REGISTERS, INT_PARAM_REGISTERS},
    value::{FloatValue, IntValue, Offset, Pointer, Size, Value},
};

#[derive(Debug, Default)]
pub struct Compiler<'src> {
    function_body: Vec<Instruction>,
    used_registers: Vec<IntRegister>,
    used_float_registers: Vec<FloatRegister>,
    /// Maps variable names to `Option<Pointer>`, or `None` when of type `()` or `!`
    scopes: Vec<HashMap<&'src str, Option<Variable>>>,
    frame_size: i64,
    requires_frame: bool,

    exports: Vec<Instruction>,
    text_section: Vec<Instruction>,

    //////// .data section ////////
    quad_globals: Vec<Instruction>,
    // long_globals: Vec<Instruction>,
    // short_globals: Vec<Instruction>,
    byte_globals: Vec<Instruction>,

    //////// .rodata section ////////
    octa_constants: Vec<Instruction>,
    quad_constants: Vec<Instruction>,
    // long_constants: Vec<Instruction>,
    // short_constants: Vec<Instruction>,
    // byte_constants: Vec<Instruction>,
}

#[derive(Debug, Clone)]
struct Variable {
    ptr: Pointer,
    kind: VariableKind,
}

#[derive(Debug, Clone, Copy)]
enum VariableKind {
    Int,
    Float,
}

impl<'src> Compiler<'src> {
    pub fn new() -> Self {
        Self {
            // start with empty global scope
            scopes: vec![HashMap::new()],
            ..Default::default()
        }
    }

    pub fn compile(mut self, tree: AnalyzedProgram<'src>) -> String {
        self.program(tree);

        let mut buf = vec![Instruction::IntelSyntax];
        buf.append(&mut self.exports);

        buf.push(Instruction::Section(Section::Text));
        buf.append(&mut self.text_section);

        if !self.quad_globals.is_empty() || !self.byte_globals.is_empty() {
            buf.push(Instruction::Section(Section::Data));
            buf.append(&mut self.quad_globals);
            // buf.append(&mut self.long_globals);
            // buf.append(&mut self.short_globals);
            buf.append(&mut self.byte_globals);
        }

        if !self.octa_constants.is_empty() || !self.quad_constants.is_empty() {
            buf.push(Instruction::Section(Section::ReadOnlyData));
            buf.append(&mut self.octa_constants);
            buf.append(&mut self.quad_constants);
            // buf.append(&mut self.long_constants);
            // buf.append(&mut self.short_constants);
            // buf.append(&mut self.byte_constants);
        }

        buf.into_iter().map(|instr| instr.to_string()).collect()
    }

    fn align(ptr: &mut i64, size: Size) {
        if *ptr % size.byte_count() != 0 {
            *ptr += size.byte_count() - *ptr % size.byte_count();
        }
    }

    fn curr_scope(&mut self) -> &mut HashMap<&'src str, Option<Variable>> {
        self.scopes.last_mut().unwrap()
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) -> Option<HashMap<&'src str, Option<Variable>>> {
        self.scopes.pop()
    }

    fn get_var(&self, name: &'src str) -> &Option<Variable> {
        for scope in self.scopes.iter().rev() {
            if let Some(ptr) = scope.get(name) {
                return ptr;
            }
        }
        unreachable!("the analyzer guarantees valid variable references");
    }

    fn add_var(&mut self, name: &'src str, size: Size, value: Value) {
        // add padding for correct alignment
        Self::align(&mut self.frame_size, size);

        // reserve space in stack frame
        self.frame_size += size.byte_count();

        // get pointer to location in stack
        let ptr = Pointer::new(size, IntRegister::Rbp, Offset::Immediate(-self.frame_size));

        // mov value onto stack
        let kind;
        self.function_body.push(match value {
            Value::Int(value) => {
                kind = VariableKind::Int;
                Instruction::Mov(ptr.clone().into(), value)
            }
            Value::Float(value) => {
                kind = VariableKind::Float;
                Instruction::Movsd(ptr.clone().into(), value)
            }
        });

        // save pointer in scope
        self.curr_scope().insert(name, Some(Variable { ptr, kind }));

        // function requires a stack frame now
        self.requires_frame = true;
    }

    fn get_free_register(&mut self) -> IntRegister {
        let next = self
            .used_registers
            .last()
            .map_or(IntRegister::Rax, |reg| reg.next());
        self.used_registers.push(next);
        next
    }

    fn get_free_float_register(&mut self) -> FloatRegister {
        let next = self
            .used_float_registers
            .last()
            .map_or(FloatRegister::Xmm0, |reg| reg.next());
        self.used_float_registers.push(next);
        next
    }

    /////////////////////////////////////////

    fn program(&mut self, node: AnalyzedProgram<'src>) {
        self.main_fn(node.main_fn);

        for func in node.functions.into_iter().filter(|func| func.used) {
            self.function_definition(func);
        }
    }

    fn main_fn(&mut self, body: AnalyzedBlock<'src>) {
        self.text_section.push(Instruction::Symbol("_start".into()));
        self.exports.push(Instruction::Global("_start".into()));

        self.function_body(body);

        self.text_section
            .push(Instruction::Mov(IntRegister::Rax.into(), 0.into()));
        self.text_section
            .push(Instruction::Call("exit".to_string()));
    }

    fn function_definition(&mut self, node: AnalyzedFunctionDefinition<'src>) {
        self.text_section
            .push(Instruction::Symbol(format!("main..{}", node.name)));

        self.push_scope();
        let mut int_param_index = 0;
        let mut float_param_index = 0;
        let mut memory_offset = 16;
        for param in node.params {
            match (
                param.type_,
                param.type_.try_into(),
                INT_PARAM_REGISTERS.get(int_param_index),
                FLOAT_PARAM_REGISTERS.get(float_param_index),
            ) {
                (Type::Float, Ok(size), _, Some(reg)) => {
                    self.add_var(param.name, size, Value::Float(FloatValue::Register(*reg)));
                    float_param_index += 1;
                }
                (_, Ok(size), None, _) | (_, Ok(size), _, None) => {
                    // add padding for correct alignment
                    Self::align(&mut memory_offset, size);

                    // save pointer in scope
                    self.curr_scope().insert(
                        param.name,
                        Some(Variable {
                            ptr: Pointer::new(size, IntRegister::Rbp, memory_offset.into()),
                            kind: match param.type_ == Type::Float {
                                true => VariableKind::Float,
                                false => VariableKind::Int,
                            },
                        }),
                    );

                    // add param size to memory offset
                    memory_offset += size.byte_count();

                    // function requires a stack frame now
                    self.requires_frame = true;
                }
                (_, Ok(size), Some(reg), _) => {
                    self.add_var(
                        param.name,
                        size,
                        Value::Int(IntValue::Register(reg.in_size(size))),
                    );
                    int_param_index += 1;
                }
                _ => {
                    self.curr_scope().insert(param.name, None);
                }
            }
        }

        self.function_body(node.block);
        self.pop_scope();

        // return to caller
        self.text_section.push(Instruction::Ret);
    }

    fn function_body(&mut self, node: AnalyzedBlock<'src>) {
        self.requires_frame = false;

        for stmt in node.stmts {
            self.statement(stmt);
            debug_assert_eq!(self.used_registers, []);
            debug_assert_eq!(self.used_float_registers, []);
        }

        if let Some(expr) = node.expr {
            match self.expression(expr) {
                Some(Value::Int(IntValue::Register(reg))) => {
                    // we expect the result to already be in %rax
                    // TODO: is that correct?
                    debug_assert_eq!(reg.qword_variant(), IntRegister::Rax);
                    debug_assert_eq!(self.used_registers, [IntRegister::Rax]);
                }
                Some(Value::Int(IntValue::Ptr(ptr))) => self.function_body.push(Instruction::Mov(
                    IntRegister::Rax.in_size(ptr.size).into(),
                    ptr.into(),
                )),
                Some(Value::Int(IntValue::Immediate(num))) => self
                    .function_body
                    .push(Instruction::Mov(IntRegister::Rax.into(), num.into())),

                Some(Value::Float(FloatValue::Register(reg))) => {
                    // we expect the result to already be in %xmm0
                    // TODO: is that correct?
                    debug_assert_eq!(reg, FloatRegister::Xmm0);
                    debug_assert_eq!(self.used_float_registers, [FloatRegister::Xmm0]);
                }
                Some(Value::Float(FloatValue::Ptr(ptr))) => self
                    .function_body
                    .push(Instruction::Movsd(FloatRegister::Xmm0.into(), ptr.into())),
                None => {}
            };
        }

        // clear used registers
        self.used_registers.clear();
        self.used_float_registers.clear();

        // get and reset frame size
        let mut frame_size = self.frame_size;
        self.frame_size = 0;

        // align frame size to 16 bytes
        Self::align(&mut frame_size, Size::Oword);

        // prologue
        if self.requires_frame || frame_size != 0 {
            // save base pointer
            self.text_section.push(Instruction::Push(IntRegister::Rbp));
            // set base pointer to stack pointer
            self.text_section.push(Instruction::Mov(
                IntRegister::Rbp.into(),
                IntRegister::Rsp.into(),
            ));
            if frame_size != 0 {
                // allocate space on stack
                self.text_section
                    .push(Instruction::Sub(IntRegister::Rsp.into(), frame_size.into()));
            }
        }

        // body
        self.text_section.append(&mut self.function_body);

        // epilogue
        if self.requires_frame || frame_size != 0 {
            // deallocate space on stack and restore base pointer
            self.text_section.push(Instruction::Leave);
        }
    }

    /////////////////////////////////////////

    fn statement(&mut self, node: AnalyzedStatement<'src>) {
        match node {
            AnalyzedStatement::Let(node) => self.let_stmt(node),
            AnalyzedStatement::Return(_) => todo!(),
            AnalyzedStatement::Loop(_) => todo!(),
            AnalyzedStatement::While(_) => todo!(),
            AnalyzedStatement::For(_) => todo!(),
            AnalyzedStatement::Break => todo!(),
            AnalyzedStatement::Continue => todo!(),
            AnalyzedStatement::Expr(expr) => {
                self.expression(expr);
            }
        }

        // Clear used registers
        self.used_registers.clear();
        self.used_float_registers.clear();
    }

    fn let_stmt(&mut self, node: AnalyzedLetStmt<'src>) {
        let expr_type = node.expr.result_type();
        match self.expression(node.expr) {
            Some(Value::Int(IntValue::Register(reg))) => {
                let size = expr_type.try_into().expect("value has type int, not `()`");
                self.add_var(node.name, size, Value::Int(reg.into()));
            }
            Some(Value::Int(IntValue::Immediate(num))) => {
                let size = expr_type.try_into().expect("value has type int, not `()`");
                self.add_var(node.name, size, Value::Int(num.into()));
            }
            Some(Value::Int(IntValue::Ptr(ptr))) => {
                // store ptr size
                let size = ptr.size;

                let reg = self.get_free_register().in_size(size);

                // temporarily move value from memory to register
                self.function_body
                    .push(Instruction::Mov(reg.into(), ptr.into()));

                // push variable to stack
                self.add_var(node.name, size, Value::Int(reg.into()));
            }
            Some(Value::Float(FloatValue::Register(reg))) => {
                self.add_var(node.name, Size::Dword, Value::Float(reg.into()));
            }
            Some(Value::Float(FloatValue::Ptr(ptr))) => {
                // store ptr size
                let size = ptr.size;

                let reg = self.get_free_float_register();

                // temporarily move value from memory to register
                self.function_body
                    .push(Instruction::Movsd(reg.into(), ptr.into()));

                // push variable to stack
                self.add_var(node.name, size, Value::Float(reg.into()));
            }
            None => {
                self.curr_scope().insert(node.name, None);
            }
        }
    }

    /////////////////////////////////////////

    fn expression(&mut self, node: AnalyzedExpression<'src>) -> Option<Value> {
        match node {
            AnalyzedExpression::Block(_node) => todo!(),
            AnalyzedExpression::If(_node) => todo!(),
            AnalyzedExpression::Int(num) => Some(Value::Int(IntValue::Immediate(num))),
            AnalyzedExpression::Float(num) => {
                let symbol_name = format!(".quad_constant_{}", self.quad_constants.len() / 2);
                self.quad_constants
                    .push(Instruction::Symbol(symbol_name.clone()));
                self.quad_constants.push(Instruction::QuadFloat(num));
                Some(Value::Float(FloatValue::Ptr(Pointer::new(
                    Size::Qword,
                    IntRegister::Rip,
                    Offset::Symbol(symbol_name),
                ))))
            }
            AnalyzedExpression::Bool(bool) => Some(Value::Int(IntValue::Immediate(bool as i64))),
            AnalyzedExpression::Char(num) => Some(Value::Int(IntValue::Immediate(num as i64))),
            AnalyzedExpression::Ident(node) => self.ident_expr(node),
            AnalyzedExpression::Prefix(_node) => todo!(),
            AnalyzedExpression::Infix(node) => self.infix_expr(*node),
            AnalyzedExpression::Assign(_node) => todo!(),
            AnalyzedExpression::Call(_node) => todo!(),
            AnalyzedExpression::Cast(_node) => todo!(),
            AnalyzedExpression::Grouped(node) => self.expression(*node),
        }
    }

    fn ident_expr(&mut self, node: AnalyzedIdentExpr<'src>) -> Option<Value> {
        self.get_var(node.ident).clone().map(|var| match var.kind {
            VariableKind::Int => Value::Int(var.ptr.into()),
            VariableKind::Float => Value::Float(var.ptr.into()),
        })
    }

    fn infix_expr(&mut self, node: AnalyzedInfixExpr<'src>) -> Option<Value> {
        let lhs = self.expression(node.lhs);
        let rhs = self.expression(node.rhs);

        match (lhs, rhs, node.op) {
            // `None` means a value of type `!` in this context, so don't do anything, as this
            // expression is unreachable at runtime
            (None, _, _) | (_, None, _) => None,
            (
                Some(Value::Int(left)),
                Some(Value::Int(right)),
                InfixOp::Plus
                | InfixOp::Minus
                | InfixOp::Mul
                | InfixOp::BitOr
                | InfixOp::BitAnd
                | InfixOp::BitXor,
            ) => {
                // if two registers were in use before, one can be freed afterwards
                let pop_rhs_reg = matches!(
                    (&left, &right),
                    (IntValue::Register(_), IntValue::Register(_))
                );

                let (left, right) = match (left, right) {
                    // when one side is already a register, use that
                    (left @ IntValue::Register(_), right @ IntValue::Register(_))
                    | (left @ IntValue::Register(_), right @ IntValue::Ptr(_))
                    | (left @ IntValue::Register(_), right @ IntValue::Immediate(_))
                    // note the swapped sides here
                    | (right @ IntValue::Ptr(_), left @ IntValue::Register(_))
                    // note the swapped sides here
                    | (right @ IntValue::Immediate(_), left @ IntValue::Register(_)) => {
                        (left, right)
                    }
                    // else move the left value into a free register and use that
                    (left @ IntValue::Ptr(_), right) | (left @ IntValue::Immediate(_), right) => {
                        let reg = self.get_free_register();
                        self.function_body.push(Instruction::Mov(reg.into(), left));
                        (reg.into(), right)
                    }
                };

                self.function_body.push(match node.op {
                    InfixOp::Plus => Instruction::Add(left.clone(), right),
                    InfixOp::Minus => Instruction::Sub(left.clone(), right),
                    InfixOp::Mul => Instruction::Imul(left.clone(), right),
                    InfixOp::BitOr => Instruction::Or(left.clone(), right),
                    InfixOp::BitAnd => Instruction::And(left.clone(), right),
                    InfixOp::BitXor => Instruction::Xor(left.clone(), right),
                    _ => unreachable!("this arm only matches with above ops"),
                });
                if pop_rhs_reg {
                    // free the rhs register
                    self.used_registers.pop();
                }
                Some(Value::Int(left))
            }
            _ => todo!(),
        }
    }
}

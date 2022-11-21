use std::{collections::HashMap, mem};

use rush_analyzer::{ast::*, InfixOp, Type};

use crate::{
    instruction::{Instruction, Section},
    register::{FloatRegister, IntRegister, FLOAT_PARAM_REGISTERS, INT_PARAM_REGISTERS},
    value::{FloatValue, IntValue, Offset, Pointer, Size, Value},
};

const BUILTIN_FUNCS: &[&str] = &["exit"];

#[derive(Debug, Default)]
pub struct Compiler<'src> {
    function_body: Vec<Instruction>,
    used_registers: Vec<IntRegister>,
    used_float_registers: Vec<FloatRegister>,
    /// Whether the compiler is currently inside function args, `true` when not `0`. Stored as a
    /// counter for nested calls.
    in_args: usize,
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
    /// Constants with 128-bits, saved as `[symbol, value]`
    octa_constants: Vec<[Instruction; 2]>,
    /// Constants with 64-bits, saved as `[symbol, value]`
    quad_constants: Vec<[Instruction; 2]>,
    // /// Constants with 32-bits, saved as `[symbol, value]`
    // long_constants: Vec<[Instruction; 2]>,
    // /// Constants with 16-bits, saved as `[symbol, value]`
    // short_constants: Vec<[Instruction; 2]>,
    // /// Constants with 8-bits, saved as `[symbol, value]`
    // byte_constants: Vec<[Instruction; 2]>,
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
            buf.extend(self.octa_constants.into_iter().flatten());
            buf.extend(self.quad_constants.into_iter().flatten());
            // buf.extend(self.long_constants.into_iter().flatten());
            // buf.extend(self.short_constants.into_iter().flatten());
            // buf.extend(self.byte_constants.into_iter().flatten());
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
        let next = self.used_registers.last().map_or(
            if self.in_args != 0 {
                IntRegister::Rdi
            } else {
                IntRegister::Rax
            },
            |reg| reg.next(),
        );
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
                    debug_assert_eq!(reg.in_qword_size(), IntRegister::Rax);
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
                let symbol_name = match self
                    .quad_constants
                    .iter()
                    .find(|[_name, value]| value == &Instruction::QuadFloat(num))
                {
                    // when a constant with the same value already exists, reuse it
                    Some([Instruction::Symbol(name), _]) => name.clone(),
                    // else create a new one
                    _ => {
                        let name = format!(".quad_constant_{}", self.quad_constants.len());
                        self.quad_constants.push([
                            Instruction::Symbol(name.clone()),
                            Instruction::QuadFloat(num),
                        ]);
                        name
                    }
                };
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
            AnalyzedExpression::Call(node) => self.call_expr(*node),
            AnalyzedExpression::Cast(node) => self.cast_expr(*node),
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
            (
                Some(Value::Float(left)),
                Some(Value::Float(right)),
                InfixOp::Plus | InfixOp::Minus | InfixOp::Mul | InfixOp::Div,
            ) => {
                // if two registers were in use before, one can be freed afterwards
                let pop_rhs_reg = matches!(
                    (&left, &right),
                    (FloatValue::Register(_), FloatValue::Register(_))
                );

                let (left, right) = match (left, right) {
                    // when one side is already a register, use that
                    (left @ FloatValue::Register(_), right @ (FloatValue::Register(_) | FloatValue::Ptr(_)))
                    // note the swapped sides here
                    | (right @ FloatValue::Ptr(_), left @ FloatValue::Register(_)) => {
                        (left, right)
                    }
                    // else move the left value into a free register and use that
                    (left @ FloatValue::Ptr(_), right) => {
                        let reg = self.get_free_float_register();
                        self.function_body.push(Instruction::Movsd(reg.into(), left));
                        (reg.into(), right)
                    }
                };

                self.function_body.push(match node.op {
                    InfixOp::Plus => Instruction::Addsd(left.clone(), right),
                    InfixOp::Minus => Instruction::Subsd(left.clone(), right),
                    InfixOp::Mul => Instruction::Mulsd(left.clone(), right),
                    InfixOp::Div => Instruction::Divsd(left.clone(), right),
                    _ => unreachable!("this arm only matches with above ops"),
                });
                if pop_rhs_reg {
                    // free the rhs register
                    self.used_registers.pop();
                }
                Some(Value::Float(left))
            }
            _ => todo!(),
        }
    }

    fn call_expr(&mut self, node: AnalyzedCallExpr<'src>) -> Option<Value> {
        let prev_used_registers = mem::take(&mut self.used_registers);
        let prev_used_float_registers = mem::take(&mut self.used_float_registers);

        // save currently used caller-saved registers on stack
        for reg in prev_used_registers
            .iter()
            .filter(|reg| reg.is_caller_saved())
        {
            self.function_body.push(Instruction::Push(*reg));
        }
        if !prev_used_float_registers.is_empty() {
            self.function_body.push(Instruction::Sub(
                IntRegister::Rsp.into(),
                (prev_used_float_registers.len() as i64 * 8).into(),
            ));
        }
        for (index, reg) in prev_used_float_registers.iter().rev().enumerate() {
            self.function_body.push(Instruction::Movsd(
                Pointer::new(Size::Qword, IntRegister::Rsp, (index as i64 * 8).into()).into(),
                (*reg).into(),
            ));
        }

        self.in_args += 1;

        // compile arg exprs
        // TODO: stack args
        for arg in node.args {
            match self.expression(arg) {
                Some(Value::Int(IntValue::Register(_)) | Value::Float(FloatValue::Register(_)))
                | None => {}
                Some(Value::Int(src)) => {
                    let reg = self.get_free_register();
                    self.function_body.push(Instruction::Mov(reg.into(), src));
                }
                Some(Value::Float(src)) => {
                    let reg = self.get_free_float_register();
                    self.function_body.push(Instruction::Movsd(reg.into(), src));
                }
            }
        }

        // call function
        self.function_body.push(Instruction::Call(
            match BUILTIN_FUNCS.contains(&node.func) {
                true => node.func.to_string(),
                false => format!("main..{}", node.func),
            },
        ));

        // move result to free register
        self.in_args -= 1;
        self.used_registers = prev_used_registers.clone();
        self.used_float_registers = prev_used_float_registers.clone();
        let result_reg = match node.result_type {
            Type::Unit | Type::Never => None,
            Type::Int | Type::Char | Type::Bool => {
                let size =
                    Size::try_from(node.result_type).expect("int, char and bool have a size");
                let reg = self.get_free_register().in_size(size);
                self.function_body.push(Instruction::Mov(
                    reg.into(),
                    IntRegister::Rax.in_size(size).into(),
                ));
                Some(Value::Int(reg.into()))
            }
            Type::Float => {
                let reg = self.get_free_float_register();
                self.function_body
                    .push(Instruction::Movsd(reg.into(), FloatRegister::Xmm0.into()));
                Some(Value::Float(reg.into()))
            }
            Type::Unknown => unreachable!("the analyzer guarantees one of the above to match"),
        };

        // restore previously used caller-saved registers from stack
        for (index, reg) in prev_used_float_registers.iter().rev().enumerate().rev() {
            self.function_body.push(Instruction::Movsd(
                (*reg).into(),
                Pointer::new(Size::Qword, IntRegister::Rsp, (index as i64 * 8).into()).into(),
            ));
        }
        if !prev_used_float_registers.is_empty() {
            self.function_body.push(Instruction::Add(
                IntRegister::Rsp.into(),
                (prev_used_float_registers.len() as i64 * 8).into(),
            ));
        }
        for reg in prev_used_registers
            .iter()
            .filter(|reg| reg.is_caller_saved())
            .rev()
        {
            self.function_body.push(Instruction::Pop(*reg));
        }

        result_reg
    }

    fn cast_expr(&mut self, node: AnalyzedCastExpr<'src>) -> Option<Value> {
        let expr = self.expression(node.expr);
        match (expr, node.type_) {
            (None, _) => None,
            (Some(Value::Int(val)), Type::Int) => Some(Value::Int(val)),
            (Some(Value::Int(_val)), Type::Float) => todo!(),
            (Some(Value::Int(_val)), Type::Bool) => todo!(),
            (Some(Value::Int(_val)), Type::Char) => todo!(),
            (Some(Value::Float(val)), Type::Int) => {
                let reg = self.get_free_register().in_qword_size();
                if let FloatValue::Register(reg) = val {
                    // TODO: .iter().position() slow?
                    self.used_float_registers.remove(
                        self.used_float_registers
                            .iter()
                            .position(|r| *r == reg)
                            .expect("returned registers should be marked as used"),
                    );
                }
                self.function_body
                    .push(Instruction::Cvttsd2si(reg.into(), val));
                Some(Value::Int(reg.into()))
            }
            (Some(Value::Float(val)), Type::Float) => Some(Value::Float(val)),
            (Some(Value::Float(_val)), Type::Bool) => todo!(),
            (Some(Value::Float(_val)), Type::Char) => todo!(),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

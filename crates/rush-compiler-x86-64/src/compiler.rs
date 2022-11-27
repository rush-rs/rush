use std::{collections::HashMap, hash::Hash, mem};

use rush_analyzer::{ast::*, InfixOp, PrefixOp, Type};

use crate::{
    instruction::{Condition, Instruction, Section},
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
    /// Internal stack pointer, separate from `%rsp`, used for pushing and popping inside stack
    /// fraame. Relative to `%rbp`, always positive.
    stack_pointer: i64,
    frame_size: i64,
    requires_frame: bool,

    exports: Vec<Instruction>,
    text_section: Vec<Instruction>,

    //////// .data section ////////
    quad_globals: Vec<(String, u64)>,
    quad_float_globals: Vec<(String, f64)>,
    // long_globals: Vec<(String, u32)>,
    // short_globals: Vec<(String, u16)>,
    byte_globals: Vec<(String, u8)>,

    //////// .rodata section ////////
    /// Constants with 128-bits
    octa_constants: HashMap<u128, String>,
    /// Constants with 64-bits
    quad_constants: HashMap<u64, String>,
    /// Constant floats with 64-bits
    quad_float_constants: HashMap<u64, String>,
    // /// Constants with 32-bits
    // long_constants: HashMap<u32, String>,
    /// Constants with 16-bits
    short_constants: HashMap<u16, String>,
    // /// Constants with 8-bits
    // byte_constants: HashMap<u8, String>,
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

        // use intel syntax
        let mut buf = vec![Instruction::IntelSyntax];
        // exports
        buf.append(&mut self.exports);

        // text section
        buf.push(Instruction::Section(Section::Text));
        buf.append(&mut self.text_section);

        // mutable globals
        buf.push(Instruction::Section(Section::Data));
        buf.extend(
            self.quad_globals
                .into_iter()
                .flat_map(|(name, value)| [Instruction::Symbol(name), Instruction::QuadInt(value)]),
        );
        buf.extend(
            self.quad_float_globals
                .into_iter()
                .flat_map(|(name, value)| {
                    [
                        Instruction::Symbol(name),
                        Instruction::QuadFloat(value.to_bits()),
                    ]
                }),
        );
        // buf.extend(
        //     self.long_globals
        //         .into_iter()
        //         .flat_map(|(name, value)| [Instruction::Symbol(name), Instruction::Long(value)]),
        // );
        // buf.extend(
        //     self.short_globals
        //         .into_iter()
        //         .flat_map(|(name, value)| [Instruction::Symbol(name), Instruction::Short(value)]),
        // );
        buf.extend(
            self.byte_globals
                .into_iter()
                .flat_map(|(name, value)| [Instruction::Symbol(name), Instruction::Byte(value)]),
        );

        // constants
        buf.push(Instruction::Section(Section::ReadOnlyData));
        buf.extend(
            self.octa_constants
                .into_iter()
                .flat_map(|(value, name)| [Instruction::Symbol(name), Instruction::Octa(value)]),
        );
        buf.extend(
            self.quad_constants
                .into_iter()
                .flat_map(|(value, name)| [Instruction::Symbol(name), Instruction::QuadInt(value)]),
        );
        buf.extend(
            self.quad_float_constants
                .into_iter()
                .flat_map(|(value, name)| {
                    [Instruction::Symbol(name), Instruction::QuadFloat(value)]
                }),
        );
        // buf.extend(
        //     self.long_constants
        //         .into_iter()
        //         .flat_map(|(value, name)| [Instruction::Symbol(name), Instruction::Long(value)]),
        // );
        buf.extend(
            self.short_constants
                .into_iter()
                .flat_map(|(value, name)| [Instruction::Symbol(name), Instruction::Short(value)]),
        );
        // buf.extend(
        //     self.byte_constants
        //         .into_iter()
        //         .flat_map(|(value, name)| [Instruction::Symbol(name), Instruction::Byte(value)]),
        // );

        buf.into_iter().map(|instr| instr.to_string()).collect()
    }

    fn align(ptr: &mut i64, size: Size) {
        if *ptr % size.byte_count() != 0 {
            *ptr += size.byte_count() - *ptr % size.byte_count();
        }
    }

    fn add_constant<T: Eq + Hash>(
        map: &mut HashMap<T, String>,
        value: T,
        size: Size,
        extra_len: usize,
    ) -> String {
        match map.get(&value) {
            // when a constant with the same value already exists, reuse it
            Some(name) => name.clone(),
            // else create a new one
            None => {
                let name = format!(".{size:#}_constant_{}", map.len() + extra_len);
                map.insert(value, name.clone());
                name
            }
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

    fn add_var(&mut self, name: &'src str, size: Size, value: Value, is_param: bool) {
        let kind = match value {
            Value::Int(_) => VariableKind::Int,
            Value::Float(_) => VariableKind::Float,
        };

        // push variable
        let comment = format!(
            "{} {name} = {value}",
            if is_param { "param" } else { "let" }
        );
        let ptr = self.push_to_stack(size, value, Some(comment));

        // save pointer in scope
        self.curr_scope().insert(name, Some(Variable { ptr, kind }));

        // function requires a stack frame now
        self.requires_frame = true;
    }

    fn push_to_stack(&mut self, size: Size, value: Value, comment: Option<String>) -> Pointer {
        // add padding for correct alignment
        Self::align(&mut self.stack_pointer, size);

        // allocate space on stack
        self.stack_pointer += size.byte_count();

        // possibly expand frame size
        self.frame_size = self.frame_size.max(self.stack_pointer);

        // get pointer to location in stack
        let ptr = Pointer::new(
            size,
            IntRegister::Rbp,
            Offset::Immediate(-self.stack_pointer),
        );

        let instr = match value {
            Value::Int(value) => Instruction::Mov(ptr.clone().into(), value),
            Value::Float(value) => Instruction::Movsd(ptr.clone().into(), value),
        };
        self.function_body.push(match comment {
            Some(comment) => Instruction::Commented(Box::new(instr), comment),
            None => instr,
        });

        ptr
    }

    fn pop_from_stack(&mut self, ptr: Pointer, dest: Value, comment: Option<String>) {
        let offset = match ptr.offset {
            Offset::Immediate(offset) => -offset,
            Offset::Symbol(_) => panic!("called `Compiler::pop()` with a symbol offset in ptr"),
        };
        // assert the pointer is in the top 8 bytes of the stack
        debug_assert_eq!(ptr.base, IntRegister::Rbp);
        debug_assert!(offset <= self.stack_pointer && offset > self.stack_pointer - 8);

        // free memory on stack
        self.stack_pointer = offset - ptr.size.byte_count();

        // move from pointer to dest
        let instr = match dest {
            Value::Int(dest) => Instruction::Mov(dest, ptr.into()),
            Value::Float(dest) => Instruction::Movsd(dest, ptr.into()),
        };
        self.function_body.push(match comment {
            Some(comment) => Instruction::Commented(Box::new(instr), comment),
            None => instr,
        });
    }

    fn get_free_register(&mut self, size: Size) -> IntRegister {
        let next = self
            .used_registers
            .last()
            .map_or(
                if self.in_args != 0 {
                    IntRegister::Rdi
                } else {
                    IntRegister::Rax
                },
                |reg| reg.next(),
            )
            .in_size(size);
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

    fn spill_int(&mut self, reg: IntRegister) -> Pointer {
        self.push_to_stack(
            reg.size(),
            Value::Int(reg.into()),
            Some(format!("{} byte spill: {reg}", reg.size().byte_count())),
        )
    }

    fn spill_int_if_used(&mut self, reg: IntRegister) -> Option<(Pointer, IntRegister)> {
        self.used_registers
            .iter()
            .find(|r| r.in_qword_size() == reg.in_qword_size())
            .copied()
            .map(|reg| (self.spill_int(reg), reg))
    }

    fn reload_int(&mut self, ptr: Pointer, reg: IntRegister) {
        self.pop_from_stack(
            ptr,
            Value::Int(reg.into()),
            Some(format!("{} byte reload: {reg}", reg.size().byte_count())),
        );
    }

    fn reload_int_if_used(&mut self, spill: Option<(Pointer, IntRegister)>) {
        if let Some((ptr, reg)) = spill {
            self.reload_int(ptr, reg);
        }
    }

    /////////////////////////////////////////

    fn program(&mut self, node: AnalyzedProgram<'src>) {
        for global in node.globals {
            self.global(global);
        }

        self.main_fn(node.main_fn);

        for func in node.functions.into_iter().filter(|func| func.used) {
            self.function_definition(func);
        }
    }

    fn global(&mut self, node: AnalyzedLetStmt<'src>) {
        let name = format!("main..{}", node.name);

        self.scopes[0].insert(
            node.name,
            Some(Variable {
                ptr: Pointer::new(
                    Size::try_from(node.expr.result_type())
                        .expect("the analyzer guarantees constant globals"),
                    IntRegister::Rip,
                    Offset::Symbol(name.clone()),
                ),
                kind: match node.expr.result_type() {
                    Type::Float => VariableKind::Float,
                    _ => VariableKind::Int,
                },
            }),
        );

        match node.expr {
            AnalyzedExpression::Int(num) => self.quad_globals.push((name, num as u64)),
            AnalyzedExpression::Float(num) => self.quad_float_globals.push((name, num)),
            AnalyzedExpression::Bool(bool) => self.byte_globals.push((name, bool as u8)),
            AnalyzedExpression::Char(num) => self.byte_globals.push((name, num)),
            _ => unreachable!("the analyzer guarantees constant globals"),
        }
    }

    fn main_fn(&mut self, body: AnalyzedBlock<'src>) {
        self.exports.push(Instruction::Global("_start".into()));
        self.text_section.push(Instruction::Symbol("_start".into()));
        self.text_section
            .push(Instruction::Call("main..main".into()));
        self.text_section
            .push(Instruction::Mov(IntRegister::Rax.into(), 0.into()));
        self.text_section.push(Instruction::Call("exit".into()));

        self.text_section
            .push(Instruction::Symbol("main..main".into()));
        self.function_body(body);
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
                    self.add_var(
                        param.name,
                        size,
                        Value::Float(FloatValue::Register(*reg)),
                        true,
                    );
                    float_param_index += 1;
                }
                (_, Ok(size), None, _) | (_, Ok(size), _, None) => {
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
                    memory_offset += 8;

                    // function requires a stack frame now
                    self.requires_frame = true;
                }
                (_, Ok(size), Some(reg), _) => {
                    self.add_var(
                        param.name,
                        size,
                        Value::Int(IntValue::Register(reg.in_size(size))),
                        true,
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
                    debug_assert_eq!(self.used_registers.len(), 1);
                    debug_assert_eq!(self.used_registers[0].in_qword_size(), IntRegister::Rax);
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
        self.stack_pointer = 0;

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
        self.text_section.push(Instruction::Ret);
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
                self.add_var(node.name, size, Value::Int(reg.into()), false);
            }
            Some(Value::Int(IntValue::Immediate(num))) => {
                let size = expr_type.try_into().expect("value has type int, not `()`");
                self.add_var(node.name, size, Value::Int(num.into()), false);
            }
            Some(Value::Int(IntValue::Ptr(ptr))) => {
                // store ptr size
                let size = ptr.size;

                let reg = self.get_free_register(size);

                // temporarily move value from memory to register
                self.function_body
                    .push(Instruction::Mov(reg.into(), ptr.into()));

                // push variable to stack
                self.add_var(node.name, size, Value::Int(reg.into()), false);
            }
            Some(Value::Float(FloatValue::Register(reg))) => {
                self.add_var(node.name, Size::Dword, Value::Float(reg.into()), false);
            }
            Some(Value::Float(FloatValue::Ptr(ptr))) => {
                // store ptr size
                let size = ptr.size;

                let reg = self.get_free_float_register();

                // temporarily move value from memory to register
                self.function_body
                    .push(Instruction::Movsd(reg.into(), ptr.into()));

                // push variable to stack
                self.add_var(node.name, size, Value::Float(reg.into()), false);
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
            AnalyzedExpression::Int(num) => match num > i32::MAX as i64 || num < i32::MIN as i64 {
                true => {
                    let symbol = Self::add_constant(
                        &mut self.quad_constants,
                        num as u64,
                        Size::Qword,
                        self.quad_float_constants.len(),
                    );
                    Some(Value::Int(IntValue::Ptr(Pointer::new(
                        Size::Qword,
                        IntRegister::Rip,
                        Offset::Symbol(symbol),
                    ))))
                }
                false => Some(Value::Int(IntValue::Immediate(num))),
            },
            AnalyzedExpression::Float(num) => {
                let symbol_name = Self::add_constant(
                    &mut self.quad_float_constants,
                    num.to_bits(),
                    Size::Qword,
                    self.quad_constants.len(),
                );
                Some(Value::Float(FloatValue::Ptr(Pointer::new(
                    Size::Qword,
                    IntRegister::Rip,
                    Offset::Symbol(symbol_name),
                ))))
            }
            AnalyzedExpression::Bool(bool) => Some(Value::Int(IntValue::Immediate(bool as i64))),
            AnalyzedExpression::Char(num) => Some(Value::Int(IntValue::Immediate(num as i64))),
            AnalyzedExpression::Ident(node) => self.ident_expr(node),
            AnalyzedExpression::Prefix(node) => self.prefix_expr(*node),
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

    fn prefix_expr(&mut self, node: AnalyzedPrefixExpr<'src>) -> Option<Value> {
        let expr_type = node.expr.result_type();
        let expr = self.expression(node.expr);
        match (expr, expr_type, node.op) {
            (Some(Value::Int(value)), Type::Int, PrefixOp::Neg) => match value {
                IntValue::Register(reg) => {
                    // negate the value in register
                    self.function_body.push(Instruction::Neg(reg.into()));
                    Some(Value::Int(reg.into()))
                }
                IntValue::Ptr(ptr) => {
                    // move value into free register
                    let reg = self.get_free_register(ptr.size);
                    self.function_body
                        .push(Instruction::Mov(reg.into(), ptr.into()));
                    // negate register
                    self.function_body.push(Instruction::Neg(reg.into()));
                    Some(Value::Int(reg.into()))
                }
                // return negated immediate
                IntValue::Immediate(num) => Some(Value::Int(IntValue::Immediate(-num))),
            },
            (Some(Value::Int(value)), Type::Bool, PrefixOp::Not) => match value {
                IntValue::Register(reg) => {
                    // xor value in register with 1
                    self.function_body
                        .push(Instruction::Xor(reg.into(), 1.into()));
                    Some(Value::Int(reg.into()))
                }
                IntValue::Ptr(ptr) => {
                    // move value into free register
                    let reg = self.get_free_register(ptr.size);
                    self.function_body
                        .push(Instruction::Mov(reg.into(), ptr.into()));
                    // xor register with 1
                    self.function_body
                        .push(Instruction::Xor(reg.into(), 1.into()));
                    Some(Value::Int(reg.into()))
                }
                IntValue::Immediate(num) => Some(Value::Int(IntValue::Immediate(num ^ 1))),
            },
            (Some(Value::Float(value)), Type::Float, PrefixOp::Neg) => {
                let reg = match value {
                    FloatValue::Register(reg) => reg,
                    FloatValue::Ptr(ptr) => {
                        let reg = self.get_free_float_register();
                        self.function_body
                            .push(Instruction::Movsd(reg.into(), ptr.into()));
                        reg
                    }
                };
                let negate_symbol =
                    Self::add_constant(&mut self.octa_constants, 1_u128 << 63, Size::Oword, 0);
                let negate_ptr =
                    Pointer::new(Size::Oword, IntRegister::Rip, Offset::Symbol(negate_symbol));
                self.function_body
                    .push(Instruction::Xorpd(reg.into(), negate_ptr.into()));
                Some(Value::Float(reg.into()))
            }
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }

    fn infix_expr(&mut self, node: AnalyzedInfixExpr<'src>) -> Option<Value> {
        // some ops handle compilation of the expressions themselves
        match node.op {
            // integer pow
            InfixOp::Pow => {
                return self.call_func(
                    Type::Int,
                    "__rush_internal_pow_int".into(),
                    vec![node.lhs, node.rhs],
                )
            }
            // logical AND
            InfixOp::And => todo!(),
            // logical OR
            InfixOp::Or => todo!(),
            _ => {}
        }

        // else compile the expressions
        let lhs = self.expression(node.lhs);
        let rhs = self.expression(node.rhs);

        // and the operation on them
        match (lhs, rhs, node.op) {
            // `None` means a value of type `!` in this context, so don't do anything, as this
            // expression is unreachable at runtime
            (None, _, _) | (_, None, _) => None,
            // basic arithmetic for `int` and bitwise ops for `int` and `bool`
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
                    // when one side is already a register, use that in matching size
                    (IntValue::Register(left), right)
                    // note the swapped sides here
                    | (right, IntValue::Register(left)) => {
                        (
                            match &right {
                                IntValue::Register(reg) => left.in_size(reg.size()),
                                IntValue::Ptr(ptr) => left.in_size(ptr.size),
                                IntValue::Immediate(_) => left,
                            },
                            right,
                        )
                    }
                    // else move the left value into a free register and use that
                    (left, right) => {
                        let reg = self.get_free_register(match &right {
                            IntValue::Ptr(ptr) => ptr.size,
                            _ => Size::Qword,
                        });
                        self.function_body.push(Instruction::Mov(reg.into(), left));
                        (reg, right)
                    }
                };

                self.function_body.push(match node.op {
                    InfixOp::Plus => Instruction::Add(left.into(), right),
                    InfixOp::Minus => Instruction::Sub(left.into(), right),
                    InfixOp::Mul => Instruction::Imul(left.into(), right),
                    InfixOp::BitOr => Instruction::Or(left.into(), right),
                    InfixOp::BitAnd => Instruction::And(left.into(), right),
                    InfixOp::BitXor => Instruction::Xor(left.into(), right),
                    _ => unreachable!("this arm only matches with above ops"),
                });
                if pop_rhs_reg {
                    // free the rhs register
                    self.used_registers.pop();
                }
                Some(Value::Int(left.into()))
            }
            // integer division
            (Some(Value::Int(left)), Some(Value::Int(right)), InfixOp::Div | InfixOp::Rem) => {
                // make sure the rax and rdx registers are free
                let spilled_rax = self.spill_int_if_used(IntRegister::Rax);
                let spilled_rdx = self.spill_int_if_used(IntRegister::Rdx);

                let mut pop_rhs_reg = false;
                let result_reg = match (&left, &right) {
                    (IntValue::Register(reg), IntValue::Register(_)) => {
                        pop_rhs_reg = true;
                        *reg
                    }
                    (IntValue::Register(reg), _) | (_, IntValue::Register(reg)) => *reg,
                    _ => self.get_free_register(Size::Qword),
                };

                // move lhs result into rax
                // analyzer guarantees `left` and `right` to be 8 bytes in size
                self.function_body
                    .push(Instruction::Mov(IntRegister::Rax.into(), left));

                // get source operand
                let source = match right {
                    right @ IntValue::Ptr(_) => right,
                    right => {
                        // move rhs result into free register
                        let mut reg = self
                            .used_registers
                            .last()
                            .map_or(IntRegister::Rdi, |reg| reg.next());
                        // don't allow %rdx
                        if reg == IntRegister::Rdx {
                            reg = reg.next()
                        }
                        self.function_body.push(Instruction::Mov(reg.into(), right));
                        reg.into()
                    }
                };

                // free the rhs register
                if pop_rhs_reg {
                    self.used_registers.pop();
                }

                // sign-extend lhs to 128 bits (required for IDIV)
                self.function_body.push(Instruction::Cqo);

                // divide
                self.function_body.push(Instruction::Idiv(source));

                // move result into result register
                self.function_body.push(Instruction::Mov(
                    result_reg.into(),
                    // use either `%rax` or `%rdx` register, depending on operator
                    IntValue::Register(match node.op {
                        InfixOp::Div => IntRegister::Rax,
                        InfixOp::Rem => IntRegister::Rdx,
                        _ => unreachable!("this arm only matches with `/` or `%`"),
                    }),
                ));

                // reload spilled registers
                self.reload_int_if_used(spilled_rax);
                self.reload_int_if_used(spilled_rdx);

                Some(Value::Int(IntValue::Register(result_reg)))
            }
            // integer shifts
            (Some(Value::Int(left)), Some(Value::Int(right)), InfixOp::Shl | InfixOp::Shr) => {
                // make sure the %rcx register is free
                let spilled_rcx = self.spill_int_if_used(IntRegister::Rcx);

                let rhs = match right {
                    IntValue::Immediate(num) => num.min(255).into(),
                    right => {
                        // if rhs is a register, free it
                        if let IntValue::Register(_) = right {
                            self.used_registers.pop();
                        }

                        // move rhs into %rcx
                        self.function_body
                            .push(Instruction::Mov(IntRegister::Rcx.into(), right));

                        // compare %rcx to 255
                        self.function_body
                            .push(Instruction::Cmp(IntRegister::Rcx.into(), 255.into()));

                        // if rhs is > 255 saturate at 255
                        let const_255_name =
                            Self::add_constant(&mut self.short_constants, 255, Size::Word, 0);
                        self.function_body.push(Instruction::Cmov(
                            Condition::Greater,
                            IntRegister::Cx.into(),
                            Pointer::new(
                                Size::Word,
                                IntRegister::Rip,
                                Offset::Symbol(const_255_name),
                            )
                            .into(),
                        ));
                        IntRegister::Cl.into()
                    }
                };

                let lhs = match left {
                    // when left side already uses a register, use that register
                    IntValue::Register(reg) => reg,
                    // else move into free one
                    left => {
                        let reg = self.get_free_register(Size::Qword);
                        self.function_body.push(Instruction::Mov(reg.into(), left));
                        reg
                    }
                };

                // shift
                self.function_body.push(match node.op {
                    InfixOp::Shl => Instruction::Shl(lhs.into(), rhs),
                    InfixOp::Shr => Instruction::Sar(lhs.into(), rhs),
                    _ => unreachable!("this arm is only entered with `<<` or `>>` operator"),
                });

                // reload spilled register
                self.reload_int_if_used(spilled_rcx);

                Some(Value::Int(IntValue::Register(lhs)))
            }
            // float arithmetic
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
                    (FloatValue::Register(left), right)
                    // note the swapped sides here
                    | (right, FloatValue::Register(left)) => {
                        (left, right)
                    }
                    // else move the left value into a free register and use that
                    (left, right) => {
                        let reg = self.get_free_float_register();
                        self.function_body.push(Instruction::Movsd(reg.into(), left));
                        (reg, right)
                    }
                };

                self.function_body.push(match node.op {
                    InfixOp::Plus => Instruction::Addsd(left.into(), right),
                    InfixOp::Minus => Instruction::Subsd(left.into(), right),
                    InfixOp::Mul => Instruction::Mulsd(left.into(), right),
                    InfixOp::Div => Instruction::Divsd(left.into(), right),
                    _ => unreachable!("this arm only matches with above ops"),
                });
                if pop_rhs_reg {
                    // free the rhs register
                    self.used_registers.pop();
                }
                Some(Value::Float(left.into()))
            }
            _ => todo!(),
        }
    }

    fn call_expr(
        &mut self,
        AnalyzedCallExpr {
            result_type,
            func,
            args,
        }: AnalyzedCallExpr<'src>,
    ) -> Option<Value> {
        self.call_func(
            result_type,
            match BUILTIN_FUNCS.contains(&func) {
                true => func.to_string(),
                false => format!("main..{func}"),
            },
            args,
        )
    }

    fn call_func(
        &mut self,
        result_type: Type,
        func: String,
        args: Vec<AnalyzedExpression<'src>>,
    ) -> Option<Value> {
        let prev_used_registers = mem::take(&mut self.used_registers);
        let prev_used_float_registers = mem::take(&mut self.used_float_registers);

        // save currently used caller-saved registers on stack
        let mut saved_register_pointers = vec![];
        let mut saved_float_register_pointers = vec![];
        for reg in prev_used_registers
            .iter()
            .filter(|reg| reg.is_caller_saved())
        {
            saved_register_pointers.push(self.spill_int(*reg));
        }
        for reg in &prev_used_float_registers {
            saved_float_register_pointers.push(self.push_to_stack(
                Size::Qword,
                Value::Float((*reg).into()),
                Some(format!("8 byte spill: {reg}")),
            ));
        }

        self.in_args += 1;

        // compile arg exprs
        let mut int_register_index = 0;
        let mut float_register_index = 0;
        let mut memory_offset = 0;
        for arg in args {
            match self.expression(arg) {
                None => {}
                Some(Value::Int(value)) => {
                    match value {
                        IntValue::Register(reg)
                            if int_register_index < INT_PARAM_REGISTERS.len() =>
                        {
                            debug_assert_eq!(
                                reg.in_qword_size(),
                                INT_PARAM_REGISTERS[int_register_index]
                            );
                        }
                        src @ (IntValue::Ptr(_) | IntValue::Immediate(_))
                            if int_register_index < INT_PARAM_REGISTERS.len() =>
                        {
                            let reg = self.get_free_register(match &src {
                                IntValue::Ptr(ptr) => ptr.size,
                                _ => Size::Qword,
                            });
                            debug_assert_eq!(
                                reg.in_qword_size(),
                                INT_PARAM_REGISTERS[int_register_index]
                            );
                            self.function_body.push(Instruction::Mov(reg.into(), src));
                        }
                        src @ (IntValue::Register(_) | IntValue::Immediate(_)) => {
                            self.function_body.push(Instruction::Mov(
                                Pointer::new(Size::Qword, IntRegister::Rsp, memory_offset.into())
                                    .into(),
                                src,
                            ));
                            memory_offset += 8;
                        }
                        IntValue::Ptr(ptr) => {
                            let reg = self.get_free_register(ptr.size);
                            self.function_body
                                .push(Instruction::Mov(reg.into(), ptr.into()));
                            self.function_body.push(Instruction::Mov(
                                Pointer::new(Size::Qword, IntRegister::Rsp, memory_offset.into())
                                    .into(),
                                reg.into(),
                            ));
                            let _popped_reg = self.used_registers.pop();
                            debug_assert_eq!(Some(reg), _popped_reg);
                            memory_offset += 8;
                        }
                    }
                    int_register_index += 1;
                }
                Some(Value::Float(value)) => {
                    match value {
                        FloatValue::Register(reg) => {
                            if float_register_index < FLOAT_PARAM_REGISTERS.len() {
                                debug_assert_eq!(reg, FLOAT_PARAM_REGISTERS[float_register_index]);
                            } else {
                                self.function_body.push(Instruction::Movsd(
                                    Pointer::new(
                                        Size::Qword,
                                        IntRegister::Rsp,
                                        memory_offset.into(),
                                    )
                                    .into(),
                                    reg.into(),
                                ));
                                memory_offset += 8;
                            }
                        }
                        FloatValue::Ptr(ptr) => {
                            if float_register_index < FLOAT_PARAM_REGISTERS.len() {
                                let reg = self.get_free_float_register();
                                debug_assert_eq!(reg, FLOAT_PARAM_REGISTERS[float_register_index]);
                                self.function_body
                                    .push(Instruction::Movsd(reg.into(), ptr.into()));
                            } else {
                                let reg = self.get_free_float_register();
                                self.function_body
                                    .push(Instruction::Movsd(reg.into(), ptr.into()));
                                self.function_body.push(Instruction::Movsd(
                                    Pointer::new(
                                        Size::Qword,
                                        IntRegister::Rsp,
                                        memory_offset.into(),
                                    )
                                    .into(),
                                    reg.into(),
                                ));
                                let _popped_reg = self.used_float_registers.pop();
                                debug_assert_eq!(Some(reg), _popped_reg);
                                memory_offset += 8;
                            }
                        }
                    }
                    float_register_index += 1;
                }
            }
        }

        // allocate the required param memory, but do not modify `self.stack_pointer`
        self.frame_size += memory_offset;

        // call function
        self.function_body.push(Instruction::Call(func));

        // move result to free register
        self.in_args -= 1;
        self.used_registers = prev_used_registers.clone();
        self.used_float_registers = prev_used_float_registers.clone();
        let result_reg = match result_type {
            Type::Unit | Type::Never => None,
            Type::Int | Type::Char | Type::Bool => {
                let size = Size::try_from(result_type).expect("int, char and bool have a size");
                let reg = self.get_free_register(size);
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
        for (reg, ptr) in prev_used_float_registers
            .into_iter()
            .zip(saved_float_register_pointers)
            .rev()
        {
            self.pop_from_stack(
                ptr,
                Value::Float(reg.into()),
                Some(format!("8 byte reload: {reg}")),
            );
        }
        for (reg, ptr) in prev_used_registers
            .into_iter()
            .filter(|reg| reg.is_caller_saved())
            .collect::<Vec<_>>() // required to know size
            .into_iter()
            .zip(saved_register_pointers)
            .rev()
        {
            self.reload_int(ptr, reg);
        }

        result_reg
    }

    fn cast_expr(&mut self, node: AnalyzedCastExpr<'src>) -> Option<Value> {
        let expr_type = node.expr.result_type();
        let expr = self.expression(node.expr);
        match (expr, expr_type, node.type_) {
            (None, _, _) => None,
            (Some(Value::Int(val)), left, right) if left == right => Some(Value::Int(val)),
            (Some(Value::Float(val)), left, right) if left == right => Some(Value::Float(val)),
            (Some(Value::Int(_val)), Type::Int, Type::Float) => todo!(),
            (Some(Value::Int(_val)), Type::Int, Type::Bool) => todo!(),
            (Some(Value::Int(_val)), Type::Int, Type::Char) => todo!(),
            (Some(Value::Float(val)), Type::Float, Type::Int) => {
                let reg = self.get_free_register(Size::Qword);
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
            (Some(Value::Float(_val)), Type::Float, Type::Bool) => todo!(),
            (Some(Value::Float(_val)), Type::Float, Type::Char) => todo!(),
            (Some(Value::Int(val)), Type::Bool | Type::Char, Type::Int) => match val {
                IntValue::Ptr(ptr) => {
                    let reg = self.get_free_register(Size::Byte);
                    self.function_body
                        .push(Instruction::Mov(reg.into(), ptr.into()));
                    Some(Value::Int(reg.in_qword_size().into()))
                }
                IntValue::Register(reg) => Some(Value::Int(reg.in_qword_size().into())),
                IntValue::Immediate(value) => Some(Value::Int((value as i64).into())),
            },
            (Some(Value::Int(_val)), Type::Bool, Type::Float) => todo!(),
            (Some(Value::Int(_val)), Type::Bool, Type::Char) => todo!(),
            (Some(Value::Int(_val)), Type::Char, Type::Float) => todo!(),
            (Some(Value::Int(_val)), Type::Char, Type::Bool) => todo!(),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

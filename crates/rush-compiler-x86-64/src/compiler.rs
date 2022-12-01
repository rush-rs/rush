use std::{collections::HashMap, hash::Hash};

use rush_analyzer::{ast::*, AssignOp, InfixOp, PrefixOp, Type};

use crate::{
    condition::Condition,
    instruction::{Instruction, Section},
    register::{FloatRegister, IntRegister, Register, FLOAT_PARAM_REGISTERS, INT_PARAM_REGISTERS},
    value::{FloatValue, IntValue, Offset, Pointer, Size, Value},
};

const BUILTIN_FUNCS: &[&str] = &["exit"];

#[derive(Debug, Default)]
pub struct Compiler<'src> {
    pub(crate) function_body: Vec<Instruction>,
    pub(crate) used_registers: Vec<IntRegister>,
    pub(crate) used_float_registers: Vec<FloatRegister>,
    /// Whether the compiler is currently inside function args, `true` when not `0`. Stored as a
    /// counter for nested calls.
    pub(crate) in_args: usize,
    /// Maps variable names to `Option<Pointer>`, or `None` when of type `()` or `!`
    pub(crate) scopes: Vec<HashMap<&'src str, Option<Variable>>>,
    /// Internal stack pointer, separate from `%rsp`, used for pushing and popping inside stack
    /// frame. Relative to `%rbp`, always positive.
    pub(crate) stack_pointer: i64,
    pub(crate) frame_size: i64,
    pub(crate) requires_frame: bool,
    pub(crate) block_count: usize,

    pub(crate) exports: Vec<Instruction>,
    pub(crate) text_section: Vec<Instruction>,

    //////// .data section ////////
    pub(crate) quad_globals: Vec<(String, u64)>,
    pub(crate) quad_float_globals: Vec<(String, f64)>,
    // pub(crate) long_globals: Vec<(String, u32)>,
    // pub(crate) short_globals: Vec<(String, u16)>,
    pub(crate) byte_globals: Vec<(String, u8)>,

    //////// .rodata section ////////
    /// Constants with 128-bits
    pub(crate) octa_constants: HashMap<u128, String>,
    /// Constants with 64-bits
    pub(crate) quad_constants: HashMap<u64, String>,
    /// Constant floats with 64-bits
    pub(crate) quad_float_constants: HashMap<u64, String>,
    // /// Constants with 32-bits
    // pub(crate) long_constants: HashMap<u32, String>,
    /// Constants with 16-bits
    pub(crate) short_constants: HashMap<u16, String>,
    // /// Constants with 8-bits
    // pub(crate) byte_constants: HashMap<u8, String>,
}

#[derive(Debug, Clone)]
pub(crate) struct Variable {
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
            block_count: usize::MAX,
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

    pub(crate) fn align(ptr: &mut i64, size: Size) {
        if *ptr % size.byte_count() != 0 {
            *ptr += size.byte_count() - *ptr % size.byte_count();
        }
    }

    pub(crate) fn add_constant<T: Eq + Hash>(
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

    pub(crate) fn new_block(&mut self) -> String {
        self.block_count = self.block_count.wrapping_add(1);
        format!(".block_{}", self.block_count)
    }

    pub(crate) fn curr_scope(&mut self) -> &mut HashMap<&'src str, Option<Variable>> {
        self.scopes.last_mut().unwrap()
    }

    pub(crate) fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub(crate) fn pop_scope(&mut self) -> Option<HashMap<&'src str, Option<Variable>>> {
        self.scopes.pop()
    }

    pub(crate) fn get_var(&self, name: &'src str) -> &Option<Variable> {
        for scope in self.scopes.iter().rev() {
            if let Some(ptr) = scope.get(name) {
                return ptr;
            }
        }
        unreachable!("the analyzer guarantees valid variable references");
    }

    pub(crate) fn add_var(&mut self, name: &'src str, size: Size, value: Value, is_param: bool) {
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

    pub(crate) fn push_to_stack(
        &mut self,
        size: Size,
        value: Value,
        comment: Option<String>,
    ) -> Pointer {
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

    pub(crate) fn pop_from_stack(&mut self, ptr: Pointer, dest: Value, comment: Option<String>) {
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

    pub(crate) fn get_tmp_register(&self, size: Size) -> IntRegister {
        self.used_registers
            .last()
            .map_or(
                if self.in_args != 0 {
                    IntRegister::Rdi
                } else {
                    IntRegister::Rax
                },
                |reg| reg.next(),
            )
            .in_size(size)
    }

    pub(crate) fn get_tmp_float_register(&self) -> FloatRegister {
        self.used_float_registers
            .last()
            .map_or(FloatRegister::Xmm0, |reg| reg.next())
    }

    pub(crate) fn get_free_register(&mut self, size: Size) -> IntRegister {
        let next = self.get_tmp_register(size);
        self.used_registers.push(next);
        next
    }

    pub(crate) fn get_free_float_register(&mut self) -> FloatRegister {
        let next = self.get_tmp_float_register();
        self.used_float_registers.push(next);
        next
    }

    pub(crate) fn spill_int(&mut self, reg: IntRegister) -> Pointer {
        self.push_to_stack(
            reg.size(),
            Value::Int(reg.into()),
            Some(format!("{} byte spill: {reg}", reg.size().byte_count())),
        )
    }

    pub(crate) fn spill_int_if_used(&mut self, reg: IntRegister) -> Option<(Pointer, IntRegister)> {
        self.used_registers
            .iter()
            .find(|r| r.in_qword_size() == reg.in_qword_size())
            .copied()
            .map(|reg| (self.spill_int(reg), reg))
    }

    pub(crate) fn reload_int(&mut self, ptr: Pointer, reg: IntRegister) {
        self.pop_from_stack(
            ptr,
            Value::Int(reg.into()),
            Some(format!("{} byte reload: {reg}", reg.size().byte_count())),
        );
    }

    pub(crate) fn reload_int_if_used(&mut self, spill: Option<(Pointer, IntRegister)>) {
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
            AnalyzedStatement::Loop(node) => self.loop_stmt(node),
            AnalyzedStatement::While(_) => todo!(),
            AnalyzedStatement::For(_) => todo!(),
            AnalyzedStatement::Break => todo!(),
            AnalyzedStatement::Continue => todo!(),
            AnalyzedStatement::Expr(node) => {
                self.expr_stmt(node);
            }
        }
    }

    fn let_stmt(&mut self, node: AnalyzedLetStmt<'src>) {
        // compile the expression
        let expr_type = node.expr.result_type();
        match self.expr_stmt(node.expr) {
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

                let reg = self.get_tmp_register(size);

                // temporarily move value from memory to register
                self.function_body
                    .push(Instruction::Mov(reg.into(), ptr.into()));

                // push variable to stack
                self.add_var(node.name, size, Value::Int(reg.into()), false);
            }
            Some(Value::Float(FloatValue::Register(reg))) => {
                self.add_var(node.name, Size::Qword, Value::Float(reg.into()), false);
            }
            Some(Value::Float(FloatValue::Ptr(ptr))) => {
                // store ptr size
                let size = ptr.size;

                let reg = self.get_tmp_float_register();

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

    fn loop_stmt(&mut self, node: AnalyzedLoopStmt<'src>) {
        let start_loop_symbol = self.new_block();
        let end_loop_symbol = self.new_block();

        self.function_body
            .push(Instruction::Symbol(start_loop_symbol.clone()));

        self.expr_stmt(AnalyzedExpression::Block(node.block.into()));
        self.function_body.push(Instruction::Jmp(start_loop_symbol));

        self.function_body
            .push(Instruction::Symbol(end_loop_symbol));
    }

    fn expr_stmt(&mut self, node: AnalyzedExpression<'src>) -> Option<Value> {
        // save current lengths of used register lists
        let used_registers_len = self.used_registers.len();
        let used_float_registers_len = self.used_float_registers.len();

        // compile the expression
        let res = self.expression(node);

        // free all newly allocated register
        self.used_registers.truncate(used_registers_len);
        self.used_float_registers.truncate(used_float_registers_len);

        res
    }

    /////////////////////////////////////////

    pub(crate) fn expression(&mut self, node: AnalyzedExpression<'src>) -> Option<Value> {
        match node {
            AnalyzedExpression::Block(node) => self.block_expr(*node),
            AnalyzedExpression::If(node) => self.if_expr(*node),
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
            AnalyzedExpression::Assign(node) => self.assign_expr(*node),
            AnalyzedExpression::Call(node) => self.call_expr(*node),
            AnalyzedExpression::Cast(node) => self.cast_expr(*node),
            AnalyzedExpression::Grouped(node) => self.expression(*node),
        }
    }

    fn block_expr(&mut self, node: AnalyzedBlock<'src>) -> Option<Value> {
        self.push_scope();
        for stmt in node.stmts {
            self.statement(stmt);
        }
        let res = node.expr.and_then(|expr| self.expression(expr));
        self.pop_scope();
        res
    }

    fn if_expr(&mut self, node: AnalyzedIfExpr<'src>) -> Option<Value> {
        let else_block_symbol = node.else_block.as_ref().map(|_| self.new_block());
        let after_if_symbol = self.new_block();
        let else_block_symbol = else_block_symbol.unwrap_or_else(|| after_if_symbol.clone());

        match Condition::try_from_expr(node.cond) {
            Ok((cond, lhs, rhs)) => {
                let lhs = self.expression(lhs)?;
                let rhs = self.expression(rhs)?;
                match (lhs, rhs) {
                    (Value::Int(lhs), Value::Int(rhs)) => {
                        if let IntValue::Register(_) = lhs {
                            self.used_registers.pop();
                        }
                        if let IntValue::Register(_) = rhs {
                            self.used_registers.pop();
                        }

                        let lhs = match (lhs, &rhs) {
                            (lhs @ IntValue::Ptr(_), IntValue::Ptr(_))
                            | (lhs @ IntValue::Immediate(_), _) => {
                                let reg = self.get_tmp_register(if let IntValue::Ptr(ptr) = &lhs {
                                    ptr.size
                                } else {
                                    Size::Qword
                                });
                                self.function_body.push(Instruction::Mov(reg.into(), lhs));
                                reg.into()
                            }
                            (lhs, _) => lhs,
                        };

                        self.function_body.push(Instruction::Cmp(lhs, rhs));
                    }
                    (Value::Float(lhs), Value::Float(rhs)) => {
                        if let FloatValue::Register(_) = lhs {
                            self.used_float_registers.pop();
                        }
                        if let FloatValue::Register(_) = rhs {
                            self.used_float_registers.pop();
                        }

                        let lhs = match lhs {
                            FloatValue::Ptr(ptr) => {
                                let reg = self.get_tmp_float_register();
                                self.function_body
                                    .push(Instruction::Movsd(reg.into(), ptr.into()));
                                reg
                            }
                            FloatValue::Register(reg) => reg,
                        };
                        self.function_body.push(Instruction::Ucomisd(lhs, rhs));

                        if cond == Condition::Equal {
                            // if floats should be equal but result is unordered, jump to
                            // else block
                            self.function_body.push(Instruction::JCond(
                                Condition::Parity,
                                else_block_symbol.clone(),
                            ));
                        } else if cond == Condition::NotEqual {
                            // if floats should not be equal and result is not unordered, jump to
                            // else block
                            self.function_body.push(Instruction::JCond(
                                Condition::NotParity,
                                else_block_symbol.clone(),
                            ));
                        }
                    }
                    _ => unreachable!("the analyzer guarantees equal types on both sides"),
                }

                self.function_body.push(Instruction::JCond(
                    cond.negated(),
                    else_block_symbol.clone(),
                ))
            }
            Err(expr) => {
                let bool = self
                    .expression(expr)?
                    .expect_int("the analyzer guarantees boolean conditions");

                match bool {
                    IntValue::Immediate(0) => {
                        self.function_body
                            .push(Instruction::Jmp(else_block_symbol.clone()));
                    }
                    IntValue::Immediate(_) => {}
                    val => {
                        self.function_body.push(Instruction::Cmp(val, 0.into()));
                        self.function_body.push(Instruction::JCond(
                            Condition::Equal,
                            else_block_symbol.clone(),
                        ));
                    }
                }
            }
        }

        let result_reg = match self.block_expr(node.then_block) {
            Some(Value::Int(val)) => match val {
                IntValue::Register(reg) => Register::Int(reg),
                _ => {
                    let reg = self.get_free_register(match &val {
                        IntValue::Ptr(ptr) => ptr.size,
                        _ => Size::Qword,
                    });
                    self.function_body.push(Instruction::Mov(reg.into(), val));
                    Register::Int(reg)
                }
            },
            Some(Value::Float(val)) => match val {
                FloatValue::Register(reg) => Register::Float(reg),
                FloatValue::Ptr(_) => {
                    let reg = self.get_free_float_register();
                    self.function_body.push(Instruction::Movsd(reg.into(), val));
                    Register::Float(reg)
                }
            },
            None => {
                self.function_body
                    .push(Instruction::Symbol(after_if_symbol));
                return None;
            }
        };
        if let Some(block) = node.else_block {
            match result_reg {
                Register::Int(_) => {
                    self.used_registers.pop();
                }
                Register::Float(_) => {
                    self.used_float_registers.pop();
                }
            }

            self.function_body
                .push(Instruction::Jmp(after_if_symbol.clone()));
            self.function_body
                .push(Instruction::Symbol(else_block_symbol));

            match self.block_expr(block) {
                Some(Value::Int(val)) => match val {
                    IntValue::Register(reg) => debug_assert_eq!(result_reg, Register::Int(reg)),
                    _ => {
                        let result_reg = result_reg
                            .expect_int("the analyzer guarantees equal types in both blocks");
                        self.used_registers.push(result_reg);
                        self.function_body
                            .push(Instruction::Mov(result_reg.into(), val))
                    }
                },
                Some(Value::Float(val)) => match val {
                    FloatValue::Register(reg) => debug_assert_eq!(result_reg, Register::Float(reg)),
                    FloatValue::Ptr(_) => {
                        let result_reg = result_reg
                            .expect_float("the analyzer guarantees equal types in both blocks");
                        // restore temporarily freed register
                        self.used_float_registers.push(result_reg);

                        self.function_body
                            .push(Instruction::Movsd(result_reg.into(), val))
                    }
                },
                None => {}
            }
        }
        self.function_body
            .push(Instruction::Symbol(after_if_symbol));

        Some(result_reg.into())
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
            (Some(Value::Int(value)), Type::Int | Type::Char, PrefixOp::Not) => {
                let reg = match value {
                    IntValue::Register(reg) => reg,
                    IntValue::Ptr(ptr) => {
                        let reg = self.get_free_register(ptr.size);
                        self.function_body
                            .push(Instruction::Mov(reg.into(), ptr.into()));
                        reg
                    }
                    IntValue::Immediate(num) => {
                        return Some(Value::Int(IntValue::Immediate(
                            match expr_type == Type::Int {
                                true => !num,
                                false => !num & 0x7F,
                            },
                        )))
                    }
                };

                self.function_body.push(Instruction::Not(reg.into()));
                Some(Value::Int(reg.into()))
            }
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
        self.compile_infix(lhs, rhs, node.op)
    }

    fn assign_expr(&mut self, node: AnalyzedAssignExpr<'src>) -> Option<Value> {
        let var = self.get_var(node.assignee).clone()?;

        if node.op == AssignOp::Pow {
            let new_val = self.call_func(
                Type::Int,
                "__rush_internal_pow_int".into(),
                vec![
                    AnalyzedExpression::Ident(AnalyzedIdentExpr {
                        result_type: Type::Int,
                        ident: node.assignee,
                    }),
                    node.expr,
                ],
            )?;
            self.function_body.push(Instruction::Mov(
                var.ptr.into(),
                new_val.expect_int("the analyzer guarantees int types for pow operations"),
            ));
            return None;
        }

        let val = self.expression(node.expr)?;

        match (val, node.op) {
            (Value::Int(IntValue::Immediate(1)), AssignOp::Plus)
            | (Value::Int(IntValue::Immediate(-1)), AssignOp::Minus) => {
                self.function_body.push(Instruction::Inc(var.ptr.into()))
            }
            (Value::Int(IntValue::Immediate(-1)), AssignOp::Plus)
            | (Value::Int(IntValue::Immediate(1)), AssignOp::Minus) => {
                self.function_body.push(Instruction::Dec(var.ptr.into()))
            }

            (Value::Int(val), AssignOp::Basic) => {
                let source = match val {
                    IntValue::Ptr(ptr) => {
                        let reg = self.get_tmp_register(ptr.size);
                        self.function_body
                            .push(Instruction::Mov(reg.into(), ptr.into()));
                        reg.into()
                    }
                    val => val,
                };
                self.function_body
                    .push(Instruction::Mov(var.ptr.into(), source));
            }
            (
                Value::Int(rhs),
                AssignOp::Plus
                | AssignOp::Minus
                | AssignOp::BitOr
                | AssignOp::BitAnd
                | AssignOp::BitXor,
            ) => {
                if matches!(rhs, IntValue::Register(_)) {
                    self.used_registers.pop();
                }
                let rhs = match rhs {
                    IntValue::Ptr(ptr) => {
                        let reg = self.get_tmp_register(ptr.size);
                        self.function_body
                            .push(Instruction::Mov(reg.into(), ptr.into()));
                        reg.into()
                    }
                    rhs => rhs,
                };
                self.function_body.push(match node.op {
                    AssignOp::Plus => Instruction::Add(var.ptr.into(), rhs),
                    AssignOp::Minus => Instruction::Sub(var.ptr.into(), rhs),
                    AssignOp::BitOr => Instruction::Or(var.ptr.into(), rhs),
                    AssignOp::BitAnd => Instruction::And(var.ptr.into(), rhs),
                    AssignOp::BitXor => Instruction::Xor(var.ptr.into(), rhs),
                    _ => unreachable!("this block is only entered with above ops"),
                });
            }
            // integer shifts
            (Value::Int(rhs), AssignOp::Shl | AssignOp::Shr) => {
                // make sure the %rcx register is free
                let spilled_rcx = self.spill_int_if_used(IntRegister::Rcx);

                let rhs = match rhs {
                    IntValue::Immediate(num) => (num & 0xff).into(),
                    rhs => {
                        // if rhs is a register, free it
                        if let IntValue::Register(_) = rhs {
                            self.used_registers.pop();
                        }

                        // move rhs into %rcx
                        self.function_body
                            .push(Instruction::Mov(IntRegister::Rcx.into(), rhs));

                        IntRegister::Cl.into()
                    }
                };

                // shift
                self.function_body.push(match node.op {
                    AssignOp::Shl => Instruction::Shl(var.ptr.into(), rhs),
                    AssignOp::Shr => Instruction::Sar(var.ptr.into(), rhs),
                    _ => unreachable!("this arm is only entered with `<<` or `>>` operator"),
                });

                // reload spilled register
                self.reload_int_if_used(spilled_rcx);
            }
            (Value::Float(val), AssignOp::Basic) => {
                let reg = match val {
                    FloatValue::Register(reg) => {
                        self.used_float_registers.pop();
                        reg
                    }
                    FloatValue::Ptr(ptr) => {
                        let reg = self.get_tmp_float_register();
                        self.function_body
                            .push(Instruction::Movsd(reg.into(), ptr.into()));
                        reg
                    }
                };
                self.function_body
                    .push(Instruction::Movsd(var.ptr.into(), reg.into()));
            }
            (rhs, op) => {
                let new_val = self.compile_infix(
                    Some(match rhs {
                        Value::Int(_) => Value::Int(var.ptr.clone().into()),
                        Value::Float(_) => Value::Float(var.ptr.clone().into()),
                    }),
                    Some(rhs),
                    match op {
                        AssignOp::Basic => unreachable!("basic assignments are covered above"),
                        AssignOp::Plus => InfixOp::Plus,
                        AssignOp::Minus => InfixOp::Minus,
                        AssignOp::Mul => InfixOp::Mul,
                        AssignOp::Div => InfixOp::Div,
                        AssignOp::Rem => InfixOp::Rem,
                        AssignOp::Pow => InfixOp::Pow,
                        AssignOp::Shl => InfixOp::Shl,
                        AssignOp::Shr => InfixOp::Shr,
                        AssignOp::BitOr => InfixOp::BitOr,
                        AssignOp::BitAnd => InfixOp::BitAnd,
                        AssignOp::BitXor => InfixOp::BitXor,
                    },
                );
                match new_val? {
                    Value::Int(val) => match val {
                        IntValue::Register(reg) => {
                            // set var to result
                            self.function_body
                                .push(Instruction::Mov(var.ptr.into(), reg.into()));
                            // free the register
                            self.used_registers.pop();
                        }
                        // TODO: validate this
                        IntValue::Ptr(_) => {
                            unreachable!("infix expressions never return pointers")
                        }
                        IntValue::Immediate(_) => {
                            unreachable!("at least one non-immediate value is always passed to `compile_infix`, so the result cannot be immediate")
                        }
                    },
                    Value::Float(val) => match val {
                        FloatValue::Register(reg) => {
                            // set var to result
                            self.function_body
                                .push(Instruction::Movsd(var.ptr.into(), reg.into()));
                            // free the register
                            self.used_float_registers.pop();
                        }
                        // TODO: validate this
                        FloatValue::Ptr(_) => {
                            unreachable!("infix expressions never return pointers")
                        }
                    },
                }
            }
        }

        None
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

    fn cast_expr(&mut self, node: AnalyzedCastExpr<'src>) -> Option<Value> {
        let expr_type = node.expr.result_type();

        match (expr_type, node.type_) {
            (Type::Int, Type::Char) => {
                return self.call_func(
                    Type::Char,
                    "__rush_internal_cast_int_to_char".into(),
                    vec![node.expr],
                )
            }
            (Type::Float, Type::Char) => {
                return self.call_func(
                    Type::Char,
                    "__rush_internal_cast_float_to_char".into(),
                    vec![node.expr],
                )
            }
            _ => {}
        }

        let expr = self.expression(node.expr);
        match (expr, expr_type, node.type_) {
            (None, _, _) => None,
            (Some(Value::Int(val)), left, right) if left == right => Some(Value::Int(val)),
            (Some(Value::Float(val)), left, right) if left == right => Some(Value::Float(val)),

            (Some(Value::Int(val)), Type::Int | Type::Char | Type::Bool, Type::Float) => {
                match val {
                    IntValue::Register(reg) => {
                        self.used_registers.pop();
                        let float_reg = self.get_free_float_register();
                        self.function_body.push(Instruction::Cvtsi2sd(
                            float_reg.into(),
                            reg.in_qword_size().into(),
                        ));
                        Some(Value::Float(float_reg.into()))
                    }
                    IntValue::Ptr(ptr) if ptr.size == Size::Qword => {
                        let reg = self.get_free_float_register();
                        self.function_body
                            .push(Instruction::Cvtsi2sd(reg.into(), ptr.into()));
                        Some(Value::Float(reg.into()))
                    }
                    IntValue::Ptr(ptr) => {
                        let reg = self.get_tmp_register(ptr.size);
                        let used_bits = ptr.size.mask();
                        self.function_body
                            .push(Instruction::Mov(reg.into(), ptr.into()));
                        self.function_body
                            .push(Instruction::And(reg.into(), used_bits.into()));

                        let float_reg = self.get_free_float_register();
                        self.function_body.push(Instruction::Cvtsi2sd(
                            float_reg.into(),
                            reg.in_dword_size().into(),
                        ));
                        Some(Value::Float(float_reg.into()))
                    }
                    IntValue::Immediate(num) => Some(Value::Float(FloatValue::Ptr(Pointer::new(
                        Size::Qword,
                        IntRegister::Rip,
                        Offset::Symbol(Self::add_constant(
                            &mut self.quad_float_constants,
                            (num as f64).to_bits(),
                            Size::Qword,
                            self.quad_constants.len(),
                        )),
                    )))),
                }
            }
            (Some(Value::Int(val)), Type::Int | Type::Char, Type::Bool) => match val {
                IntValue::Register(reg) => {
                    self.function_body
                        .push(Instruction::Cmp(reg.into(), 0.into()));
                    self.function_body.push(Instruction::SetCond(
                        Condition::NotEqual,
                        reg.in_byte_size(),
                    ));
                    Some(Value::Int(reg.into()))
                }
                val @ IntValue::Ptr(_) => {
                    self.function_body.push(Instruction::Cmp(val, 0.into()));
                    let reg = self.get_free_register(Size::Byte);
                    self.function_body
                        .push(Instruction::SetCond(Condition::NotEqual, reg));
                    Some(Value::Int(reg.into()))
                }
                IntValue::Immediate(num) => {
                    Some(Value::Int(IntValue::Immediate((num != 0) as i64)))
                }
            },
            (Some(Value::Float(val)), Type::Float, Type::Int) => {
                let reg = self.get_free_register(Size::Qword);
                if let FloatValue::Register(_) = val {
                    self.used_float_registers.pop();
                }
                self.function_body
                    .push(Instruction::Cvttsd2si(reg.into(), val));
                Some(Value::Int(reg.into()))
            }
            (Some(Value::Float(val)), Type::Float, Type::Bool) => {
                // move the value into a register
                let reg = match val {
                    FloatValue::Register(reg) => reg,
                    FloatValue::Ptr(ptr) => {
                        let reg = self.get_free_float_register();
                        self.function_body
                            .push(Instruction::Movsd(reg.into(), ptr.into()));
                        reg
                    }
                };

                // get a constant zero to compare against
                let float_zero_symbol = Self::add_constant(
                    &mut self.quad_float_constants,
                    0_f64.to_bits(),
                    Size::Qword,
                    self.quad_constants.len(),
                );

                // compare
                self.function_body.push(Instruction::Ucomisd(
                    reg,
                    FloatValue::Ptr(Pointer::new(
                        Size::Qword,
                        IntRegister::Rip,
                        Offset::Symbol(float_zero_symbol),
                    )),
                ));

                // get result
                let reg = self.get_free_register(Size::Byte);
                self.function_body
                    .push(Instruction::SetCond(Condition::NotEqual, reg));
                let parity_reg = self.get_tmp_register(Size::Byte);
                self.function_body
                    .push(Instruction::SetCond(Condition::Parity, parity_reg));
                self.function_body
                    .push(Instruction::Or(reg.into(), parity_reg.into()));

                Some(Value::Int(IntValue::Register(reg)))
            }
            (Some(Value::Int(val)), Type::Bool | Type::Char, Type::Int) => match val {
                IntValue::Ptr(ptr) => {
                    let reg = self.get_free_register(Size::Qword);
                    self.function_body
                        .push(Instruction::Mov(reg.in_size(ptr.size).into(), ptr.into()));
                    self.function_body
                        .push(Instruction::And(reg.into(), 0xff.into()));
                    Some(Value::Int(reg.into()))
                }
                IntValue::Register(reg) => {
                    self.function_body
                        .push(Instruction::And(reg.in_qword_size().into(), 0xff.into()));
                    Some(Value::Int(reg.in_qword_size().into()))
                }
                IntValue::Immediate(value) => Some(Value::Int((value as i64).into())),
            },
            (Some(Value::Int(val)), Type::Bool, Type::Char) => Some(Value::Int(val)),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

use std::{borrow::Cow, collections::HashMap, rc::Rc};

use rush_analyzer::{ast::*, AssignOp, InfixOp, PrefixOp, Type};

use crate::{
    instruction::{Block, CommentConfig, Instruction, IntRegisterPointer, Pointer},
    register::{FloatRegister, IntRegister, Register},
    utils::{DataObj, DataObjType, DivisionOutput, Function, Loop, Size, Variable},
};

pub struct Compiler<'tree> {
    /// Specifies all exported labels of the program.
    pub(crate) exports: Vec<Cow<'static, str>>,
    /// Labels and their basic blocks which contain instructions.
    pub(crate) blocks: Vec<Block<'tree>>,
    /// Maps the raw label to their count of occurrences.
    pub(crate) label_count: HashMap<&'static str, usize>,
    /// Points to the current section which is inserted to.
    pub(crate) curr_block: usize,
    /// Data section for storing mutabable global variables.
    pub(crate) data_section: Vec<DataObj>,
    /// Data section for storing float constants and immutable globals.
    pub(crate) rodata_section: Vec<DataObj>,
    /// Holds metadata about the current function
    pub(crate) curr_fn: Option<Function>,
    /// Holds metadata about the current loop(s)
    pub(crate) loops: Vec<Loop>,
    /// The last element is the current scope.
    pub(crate) scopes: Vec<HashMap<&'tree str, Variable>>,
    /// Holds the global variables of the program.
    pub(crate) globals: HashMap<&'tree str, Variable>,
    /// Specifies all currentlu used registers.
    pub(crate) used_registers: Vec<(Register, Size)>,
}

impl<'tree> Compiler<'tree> {
    /// Creates and returns a new [`Compiler`].
    pub fn new() -> Self {
        Self {
            blocks: vec![],
            label_count: HashMap::new(),
            exports: vec![],
            curr_block: 0,
            data_section: vec![],
            rodata_section: vec![],
            scopes: vec![],
            globals: HashMap::new(),
            curr_fn: None,
            loops: vec![],
            used_registers: vec![],
        }
    }

    /// Compiles the source AST into an IBM System/390 targeted Assembly program.
    pub fn compile(
        &mut self,
        ast: AnalyzedProgram<'tree>,
        comment_config: &CommentConfig,
    ) -> String {
        // define globals
        for var in ast.globals.into_iter().filter(|g| g.used) {
            self.define_global(var.name, var.mutable, var.expr)
        }

        // compile `main` fn
        self.define_main_fn(ast.main_fn);

        // compile other functions
        for func in ast.functions.into_iter().filter(|f| f.used) {
            self.function_definition(func)
        }

        // Perform trivial optimizations.
        self.optimize();

        // generate Assembly
        self.codegen(comment_config)
    }

    // Is invoked before assembly generation.
    // Performs trivial optimizations on the generated code.
    fn optimize(&mut self) {
        for block in self.blocks.iter_mut() {
            let new_instructions = block.instructions.clone().into_iter().filter(
                |(instruction, _)| !matches!(instruction, Instruction::Lgr(dest, src) if dest == src)
                ).collect();

            block.instructions = new_instructions;
        }
    }

    /// Generates the Assembly representation from the compiled program.
    /// This function is invoked in the last step of compilation and only generates the output.
    fn codegen(&self, comment_config: &CommentConfig) -> String {
        let mut output = String::new();

        // `.global` label exports
        output += &self
            .exports
            .iter()
            .map(|e| format!(".globl {e}\n"))
            .collect::<String>();

        // basic block labels with their instructions
        output += "\n.section .text\n";
        output += &self
            .blocks
            .iter()
            .map(|b| b.display(comment_config))
            .collect::<String>();

        // zero-values for globals under the `.data` section
        if !self.data_section.is_empty() {
            output += &format!(
                "\n.section .data\n{}",
                self.data_section
                    .iter()
                    .map(|d| format!("\n{d}\n"))
                    .collect::<String>()
            );
        }

        // defines constants (like floats) under the `.rodata` section
        if !self.rodata_section.is_empty() {
            output += &format!(
                "\n.section .rodata\n.align	8\n{}",
                self.rodata_section
                    .iter()
                    .map(|d| format!("\n{d}\n"))
                    .collect::<String>()
            );
        }

        output
    }

    /// Creates the `_start` label and compiles a call to the compiled `main` function.
    fn define_main_fn(&mut self, node: AnalyzedBlock<'tree>) {
        let start_label = "_start";
        self.blocks.push(Block::new(start_label.into()));
        self.exports.push(start_label.into());

        let main_label = "main..main";
        self.blocks.push(Block::new(main_label.into()));

        // call the `main` function from `_start`
        self.insert_call(main_label.into());
        // exit with code 0 by default
        self.insert(Instruction::Lghi(IntRegister::R2, 0));
        self.insert_call("exit".into());

        // add the epilogue label
        let epilogue_label = self.gen_label("epilogue");
        self.curr_fn = Some(Function::new(Rc::clone(&epilogue_label)));

        // compile the function body
        self.insert_at(main_label);
        self.push_scope();
        self.function_body(node);
        self.pop_scope();

        // insertion after body was compiled; frame size is now known
        let mut prologue = self.prologue();
        self.insert_at(main_label); // resets the current block back to the fn block
        prologue.append(&mut self.blocks[self.curr_block].instructions);
        self.blocks[self.curr_block].instructions = prologue;

        self.blocks.push(Block::new(Rc::clone(&epilogue_label)));
        self.epilogue()
    }

    /// Defines a new global variable.
    /// If the global is mutable, it is placed in the `.data` section.
    /// Otherwise, it is placed in the `.rodata` section.
    fn define_global(
        &mut self,
        label: &'tree str,
        mutable: bool,
        value: AnalyzedExpression<'tree>,
    ) {
        let type_ = value.result_type();
        let data = match (type_, value) {
            (Type::Int(0), AnalyzedExpression::Int(val)) => DataObjType::Quad(val),
            (Type::Bool(0), AnalyzedExpression::Bool(val)) => DataObjType::Byte(val as u8),
            (Type::Char(0), AnalyzedExpression::Char(val)) => DataObjType::Byte(val),
            (Type::Float(0), AnalyzedExpression::Float(val)) => DataObjType::Float(val),
            _ => unreachable!("other types cannot occur in globals"),
        };

        let value = match mutable {
            true => {
                let label = label.into();
                self.data_section.push(DataObj {
                    label: Rc::clone(&label),
                    data,
                });
                Pointer::Label(label)
            }
            false => {
                // if there is already a label with the same value, it is used
                match self.rodata_section.iter().find(|d| d.data == data) {
                    Some(DataObj { label, .. }) => Pointer::Label(Rc::clone(label)),
                    None => {
                        self.rodata_section.push(DataObj {
                            label: label.into(),
                            data,
                        });
                        Pointer::Label(label.into())
                    }
                }
            }
        };

        self.globals.insert(
            label,
            Variable {
                type_,
                value: Some(value),
            },
        );
    }

    /// Compiles an [`AnalyzedFunctionDefinition`] declaration.
    fn function_definition(&mut self, node: AnalyzedFunctionDefinition<'tree>) {
        let fn_block = format!("main..{}", node.name).into();
        self.blocks.push(Block::new(Rc::clone(&fn_block)));

        let epilogue_label = self.gen_label("epilogue");
        self.curr_fn = Some(Function::new(Rc::clone(&epilogue_label)));

        self.push_scope();
        self.insert_at(&fn_block);

        let mut param_store_instructions =
            vec![(Instruction::Comment("save params on stack".into()), None)];

        // specifies the current param
        let mut int_cnt = 0; // 0 = r0
        let mut float_cnt = 0; // 0 = f0

        // specifies the memory offset to use when params are spilled
        // is incremented in steps of 8
        let mut mem_offset = 0;

        // save all param values in the current scope / on the stack
        for param in &node.params {
            match param.type_ {
                Type::Float(0) => {
                    // match FloatRegister::nth_param(float_cnt) {
                    //     Some(reg) => {
                    //         let offset = self.get_offset(Size::Dword);
                    //
                    //         param_store_instructions.push((
                    //             Instruction::Fsd(reg, Pointer::Register(IntRegister::Fp, offset)),
                    //             Some(format!("param {} = {reg}", param.name).into()),
                    //         ));
                    //
                    //         // insert the param into the scope
                    //         self.scope_mut().insert(
                    //             param.name,
                    //             Variable {
                    //                 type_: param.type_,
                    //                 value: Some(Pointer::Register(IntRegister::Fp, offset)),
                    //             },
                    //         );
                    //     }
                    //     None => {
                    //         // if there are spilled params, insert their location into the scope
                    //         self.scope_mut().insert(
                    //             param.name,
                    //             Variable {
                    //                 type_: param.type_,
                    //                 value: Some(Pointer::Register(IntRegister::Fp, mem_offset)),
                    //             },
                    //         );
                    //         mem_offset += 8;
                    //     }
                    // }
                    // float_cnt += 1;
                    //
                    todo!("floats are not supported at this point")
                }
                Type::Int(_) | Type::Char(_) | Type::Bool(_) | Type::Float(_) => {
                    match IntRegister::nth_param(int_cnt) {
                        Some(reg) => {
                            let size = Size::from(param.type_);
                            let offset = self.get_offset(size);

                            // use `sb` or `sd` depending on the size
                            match size {
                                Size::Byte => param_store_instructions.push((
                                    Instruction::Store8(
                                        reg,
                                        IntRegisterPointer(IntRegister::R15, offset),
                                    ),
                                    Some(format!("param {} = {reg}", param.name).into()),
                                )),
                                Size::Long => todo!("impl long"),
                                Size::Quad => param_store_instructions.push((
                                    Instruction::Store64(
                                        reg,
                                        IntRegisterPointer(IntRegister::R15, offset),
                                    ),
                                    Some(format!("param {} = {reg}", param.name).into()),
                                )),
                            }

                            // insert the param into the scope
                            self.scope_mut().insert(
                                param.name,
                                Variable {
                                    type_: param.type_,
                                    value: Some(Pointer::Register(IntRegisterPointer(IntRegister::R15, offset))),
                                },
                            );
                        }
                        None => {
                            // if there are spilled params, insert their location into the scope
                            self.scope_mut().insert(
                                param.name,
                                Variable {
                                    type_: param.type_,
                                    value: Some(Pointer::Register(IntRegisterPointer(IntRegister::R15, mem_offset))),
                                },
                            );
                            mem_offset += 8;
                        }
                    }
                    int_cnt += 1;
                }
                Type::Unit | Type::Never => {
                    // insert a dummy value into the scope
                    self.scope_mut().insert(param.name, Variable::unit());
                }
                Type::Unknown => unreachable!("analyzer would have failed"),
            }
        }

        // compile the function body
        self.function_body(node.block);
        self.pop_scope();

        // compile and prepend the prologue
        let mut prologue = self.prologue();
        self.insert_at(&fn_block); // resets the current block back to the fn block
        prologue.append(&mut param_store_instructions);
        prologue.append(&mut self.blocks[self.curr_block].instructions);
        self.blocks[self.curr_block].instructions = prologue;

        // compile epilogue
        self.blocks.push(Block::new(epilogue_label));
        self.epilogue()
    }

    /// Compiles the body of a function.
    /// Does not push a new scope.
    fn function_body(&mut self, node: AnalyzedBlock<'tree>) {
        self.insert(Instruction::Comment("begin body".into()));

        // compile each statement
        for stmt in node.stmts {
            self.statement(stmt);
        }

        // places the result of the optional expression in a return value register
        // for `int`, `bool`, and `char`:   `GR2`
        // for `float`:                     `F0` TODO: what is the return register for floats?
        if let Some(expr) = node.expr {
            // if the result register does not match the desired register, insert a move instruction
            match self.expression(expr) {
                Some(Register::Int(reg)) => {
                    self.insert_movi(IntRegister::R2, reg);
                }
                Some(Register::Float(reg)) => {
                    // self.insert(Instruction::Fmv(FloatRegister::Fa0, reg));
                    todo!("implement this")
                }
                None => {} // ignore unit values
            }
        }

        self.insert(Instruction::Comment("end body".into()));
    }

    /// Compiles an [`AnalyzedBlock`].
    /// Automatically manages the scope for the block.
    fn block(&mut self, node: AnalyzedBlock<'tree>) -> Option<Register> {
        self.push_scope();

        for stmt in node.stmts {
            self.statement(stmt)
        }

        // return expr register if there is an expr
        let res = node.expr.and_then(|e| self.expression(e));

        self.pop_scope();

        res
    }

    /// Copiles an [`AnalyzedStatement`].
    /// Invokes a corresponding function for most of the statements.
    fn statement(&mut self, node: AnalyzedStatement<'tree>) {
        match node {
            AnalyzedStatement::Let(node) => self.let_statement(node),
            AnalyzedStatement::Return(node) => self.return_stmt(node),
            AnalyzedStatement::Loop(node) => self.loop_stmt(node),
            AnalyzedStatement::While(node) => self.while_stmt(node),
            AnalyzedStatement::For(node) => self.for_stmt(node),
            AnalyzedStatement::Break => self.insert_jmp(
                Rc::clone(&self.curr_loop().after_loop),
                Some("break".into()),
            ),
            AnalyzedStatement::Continue => self.insert_jmp(
                Rc::clone(&self.curr_loop().loop_head),
                Some("continue".into()),
            ),
            AnalyzedStatement::Expr(node) => {
                self.expression(node);
            }
        }
    }

    /// Compiles an [`AnalyzedReturnStmt`].
    /// If the node contains an optional expr, it is compiled and its result is moved into the
    /// correct return-value register (corresponding to the result type of the expr).
    fn return_stmt(&mut self, node: AnalyzedReturnStmt<'tree>) {
        // if there is an optional expression, use its value as the result
        if let Some(expr) = node {
            match self.expression(expr) {
                None => {}                                      // returns unit, do nothing
                Some(Register::Int(IntRegister::R2)) => {}      // already in correct register
                //Some(Register::Float(FloatRegister::Fa0)) => {} // already in correct register
                Some(Register::Int(reg)) => self.insert_movi(IntRegister::R2, reg),
                Some(Register::Float(reg)) => {
                    //self.insert(Instruction::Fmv(FloatRegister::Fa0, reg))
                    todo!("add float support")
                }
            }
        }

        // jump to the function's epilogue label
        self.insert_jmp(
            Rc::clone(&self.curr_fn().epilogue_label),
            Some("return".into()),
        );
    }

    /// Compiles an [`AnalyzedLoopStmt`].
    /// After each iteration, there is an unconditional jump back to the loop head (i.e. `continue`).
    /// In this looping construct, manual control flow like `break` is mandatory to quit the loop.
    fn loop_stmt(&mut self, node: AnalyzedLoopStmt<'tree>) {
        let loop_head = self.append_block("loop_head");
        let after_loop = self.gen_label("after_loop");

        self.loops
            .push(Loop::new(Rc::clone(&loop_head), Rc::clone(&after_loop)));

        self.insert_at(&loop_head);
        self.block(node.block);
        self.loops.pop();

        // jump back to the loop head
        self.insert_jmp(loop_head, None);

        self.blocks.push(Block::new(Rc::clone(&after_loop)));
        self.insert_at(&after_loop);
    }

    /// Compiles an [`AnalyzedWhileStmt`].
    /// Before each iteration, the loop condition is evaluated.
    /// If the result is `false`, there is a jump to the basic block after the loop (i.e. `break`).
    fn while_stmt(&mut self, node: AnalyzedWhileStmt<'tree>) {
        let while_head = self.append_block("while_head");
        let after_loop = self.gen_label("after_while");

        // compile the condition
        self.insert_at(&while_head);
        self.insert(Instruction::Comment("while condition".into()));

        // if the cond is `!`, return here
        let Some(cond) = self.expression(node.cond) else {
            return
        };

        // if the condition evaluates to `false`, break out of the loop
        self.insert(Instruction::CompareIntImm(cond.into(), false as i8));
        self.insert(Instruction::BranchEq(Rc::clone(&after_loop)));

        self.loops
            .push(Loop::new(Rc::clone(&while_head), Rc::clone(&after_loop)));

        // compile the body
        self.insert(Instruction::Comment("while body".into()));
        self.block(node.block);

        // jump back to the loop head
        self.insert_jmp(while_head, None);

        self.loops.pop();

        // place the cursor after the loop body
        self.blocks.push(Block::new(Rc::clone(&after_loop)));
        self.insert_at(&after_loop);
    }

    /// Compiles an [`AnalyzedForStmt`].
    /// Before the loop starts, an induction variable is set to a value.
    /// Before each iteration, the loop condition is checked.
    /// If the condition evaluates to `false`, there is a `break` / jump.
    /// At the end of each iteration, the update expression is executed, its result value is omitted.
    fn for_stmt(&mut self, node: AnalyzedForStmt<'tree>) {
        let for_head = self.append_block("for_head");
        let after_loop_label = self.gen_label("after_for");

        //// INIT ////
        self.insert(Instruction::Comment("for init".into()));

        let type_ = node.initializer.result_type();
        let ptr = self.save_expr_on_stack(node.initializer, node.ident.to_string());

        // add a new scope and insert the induction variable into it
        self.push_scope();

        self.scope_mut()
            .insert(node.ident, Variable { type_, value: ptr.map(|p| Pointer::Register(p)) });

        //// CONDITION ////
        self.insert_at(&for_head);

        self.insert(Instruction::Comment("for condition".into()));

        // if the cond is `!`, return here
        let Some(cond) = self.expression(node.cond) else {
            return
        };

        self.insert(Instruction::CompareIntImm(cond.into(), false as i8));
        self.insert(Instruction::BranchEq(Rc::clone(&after_loop_label)));

        //// BODY ////
        self.insert(Instruction::Comment("for body".into()));
        let for_update_label = self.gen_label("for_update");

        self.loops.push(Loop::new(
            Rc::clone(&for_update_label),
            Rc::clone(&after_loop_label),
        ));

        self.block(node.block);
        self.loops.pop();

        //// UPDATE EXPR ////
        self.blocks.push(Block::new(Rc::clone(&for_update_label)));
        self.insert_at(&for_update_label);
        self.expression(node.update);

        // jump back to `for_head`
        self.insert_jmp(for_head, None);

        //// AFTER ////
        self.pop_scope();
        self.blocks.push(Block::new(Rc::clone(&after_loop_label)));
        self.insert_at(&after_loop_label);
    }

    /// Compiles an [`AnalyzedLetStmt`]
    /// Allocates space for a new variable on the stack.
    fn let_statement(&mut self, node: AnalyzedLetStmt<'tree>) {
        let type_ = node.expr.result_type();

        // save the expression result on the stack & insert the variable into the current scope
        let ptr = self.save_expr_on_stack(node.expr, format!("let {}", node.name));
        self.scope_mut()
            .insert(node.name, Variable { type_, value: ptr.map(|p| Pointer::Register(p)) });
    }

    fn save_ireg_on_stack(&mut self, reg: IntRegister, comment: Option<String>) -> i64 {
            let offset = self.get_offset(Size::Quad);

            self.insert_with_comment(
                    Instruction::Store64(reg, IntRegisterPointer(IntRegister::R15, offset)),
                    comment.unwrap_or("".into()).into(),
            );

            offset
    }

    fn restore_ireg_from_stack(&mut self, reg: IntRegister, offset: i64) {
        self.insert_with_comment(
            Instruction::Load64(reg, IntRegisterPointer(IntRegister::R15, offset)),
            "restore after save".into(),
        )
    }

    fn save_freg_on_stack(&mut self, reg: FloatRegister, comment: Option<String>) -> i64 {
            let offset = self.get_offset(Size::Quad);

            self.insert_with_comment(
                    Instruction::StoreGeneric(reg.into(), IntRegisterPointer(IntRegister::R15, offset)),
                    comment.unwrap_or("".into()).into(),
            );

            offset
    }

    fn restore_freg_from_stack(&mut self, reg: FloatRegister, offset: i64) {
        self.insert_with_comment(
            Instruction::LoadLengthened(reg, IntRegisterPointer(IntRegister::R15, offset)),
            "restore after save".into(),
        )
    }

    fn save_expr_on_stack(
        &mut self,
        node: AnalyzedExpression<'tree>,
        comment_prefix: String,
    ) -> Option<IntRegisterPointer> {
        let type_ = node.result_type();
        let reg = self.expression(node)?;
        let comment = format!("{comment_prefix} = {reg}");
        let offset = self.get_offset(Size::from(type_));

        match reg {
            Register::Int(reg) => match type_ {
                Type::Bool(0) | Type::Char(0) => self.insert_with_comment(
                    Instruction::Store8(reg, IntRegisterPointer(IntRegister::R15, offset)),
                    comment.into(),
                ),
                Type::Int(_) | Type::Float(_) | Type::Bool(_) | Type::Char(_) => {
                    self.insert_with_comment(
                        Instruction::Store64(reg, IntRegisterPointer(IntRegister::R15, offset)),
                        comment.into(),
                    );
                }
                _ => unreachable!("only the types above use int registers"),
            },
            Register::Float(reg) => {
                self.insert_with_comment(
                Instruction::StoreGeneric(reg.into(), IntRegisterPointer(IntRegister::R15, offset)),
                comment.into(),
                );
            },
        };

        Some(IntRegisterPointer(IntRegister::R15, offset))
    }

    /// Compiles an [`AnalyzedExpression`].
    pub(crate) fn expression(&mut self, node: AnalyzedExpression<'tree>) -> Option<Register> {
        match node {
            AnalyzedExpression::Int(value) => {
                let dest_reg = self.get_int_reg();
                // Check whether the immediate is too large to fit into an LGFI instruction.
                if !(-2147483648..=2147483647).contains(&value) {
                    // Set the entire register to 0.
                    self.insert_with_comment(Instruction::Xgr(dest_reg, dest_reg), format!("{dest_reg} = 0").into());

                    // Load high 31 bits of the number.
                    const HI_31_BITS_MASK: i64 = 0x7FFFFFFF00000000;
                    let hi_bits = (value & HI_31_BITS_MASK) >> 32;

                    // The low 32 bits of the number can be loaded.
                    const LO_32_BITS_MASK: i64 = 0xFFFFFFFF;
                    let lo_bits = value & LO_32_BITS_MASK;

                    debug_assert!(lo_bits | (hi_bits << 32) == value, "{} != {value}", lo_bits | hi_bits);

                    self.insert_with_comment(
                        Instruction::Llihf(dest_reg, hi_bits as u32),
                        format!("({value} & 0x{HI_31_BITS_MASK:x}) >> 32").into(),
                    );

                    self.insert_with_comment(
                        Instruction::Oilf(dest_reg, lo_bits as u32),
                        format!("{value} & 0x{LO_32_BITS_MASK:x}").into(),
                    );
                } else {
                    self.insert_with_comment(
                        Instruction::Lgfi(dest_reg, value),
                        format!("{value}").into(),
                    );
                }

                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Bool(value) => {
                let dest_reg = self.get_int_reg();
                self.insert(Instruction::Lghi(dest_reg, value as i16));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Char(value) => {
                let dest_reg = self.get_int_reg();
                self.insert(Instruction::Lghi(dest_reg, value as i16));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Float(value) => {
                let dest_reg = self.get_float_reg();

                // if there is already a float constant with this value, use its label
                // otherwise, create a new float constant under the `.rodata` section
                let float_value_label = match self
                    .rodata_section
                    .iter()
                    .find(|o| o.data == DataObjType::Float(value))
                {
                    Some(obj) => Rc::clone(&obj.label),
                    None => {
                        let label = format!("float_constant_{}", self.rodata_section.len()).into();
                        self.rodata_section.push(DataObj {
                            label: Rc::clone(&label),
                            data: DataObjType::Float(value),
                        });
                        label
                    }
                };

                // Ensure that R0 will not be used as a base register.
                self.use_reg(IntRegister::R0.into(), Size::Quad);
                let float_addr_ireg = self.get_int_reg();
                self.release_reg(IntRegister::R0.into());

                self.insert_with_comment(
                    Instruction::LoadAddrRelativeLong(float_addr_ireg, float_value_label),
                    format!("addr of {value}").into(),
                );

                self.insert_with_comment(
                    Instruction::LoadLengthenedB(dest_reg, IntRegisterPointer(float_addr_ireg, 0)),
                    format!("load {value}").into(),
                );


                // load the value from the data label into `dest_reg`
                // self.insert(Instruction::Fld(
                //     dest_reg,
                //     Pointer::Label(float_value_label),
                // ));

                Some(Register::Float(dest_reg))
            }
            AnalyzedExpression::Ident(ident) => {
                // if this is a placeholder or dummy variable, return `None`
                let (ptr, type_) = match self.resolve_variable(ident.ident).type_ {
                    Type::Unit | Type::Unknown | Type::Never => return None,
                    type_ => {
                        (self.load_variable_from_name(ident.ident), type_)
                    },
                };
                // `clone` is okay here, since it only clones a `Rc`
                Some(self.load_value_from_pointer(ptr.clone()?, type_, ident.ident))
            }
            AnalyzedExpression::Prefix(node) => self.prefix_expr(*node),
            AnalyzedExpression::Infix(node) => self.infix_expr(*node),
            AnalyzedExpression::Assign(node) => {
                self.assign_expr(*node);
                None
            }
            AnalyzedExpression::Call(node) => self.call_expr(*node),
            AnalyzedExpression::Cast(node) => self.cast_expr(*node),
            AnalyzedExpression::Grouped(node) => self.expression(*node),
            AnalyzedExpression::Block(node) => self.block(*node),
            AnalyzedExpression::If(node) => self.if_expr(*node),
        }
    }

    /// Compiles an [`AnalyzedPrefixExpr`].
    fn prefix_expr(&mut self, node: AnalyzedPrefixExpr<'tree>) -> Option<Register> {
        let lhs_type = node.expr.result_type();

        if node.op == PrefixOp::Ref {
            if let AnalyzedExpression::Ident(ident) = node.expr {
                // let variable = self.load_variable_from_name(ident.ident);
                let pointer = self.load_variable_from_name(ident.ident);
                let dest_reg = self.get_int_reg();

                // TODO: does this handle normal pointers as well?

                // match var_ptr.clone().unwrap() {
                    // Pointer::Register(IntRegisterPointer(_, offset)) => {
                        self.insert_with_comment(
                            Instruction::Lgr(dest_reg.into(), IntRegister::R15.into()),
                            format!("&{}: copy GR15", ident.ident).into(),
                        );

                        self.insert_with_comment(
                            Instruction::Aghi(dest_reg, pointer.unwrap().offset().try_into().expect("offset too large")),
                            format!("&{}: offset", ident.ident).into(),
                        );
                    // }
                    // Pointer::Label(label) => self.insert_with_comment(
                    //     Instruction::La(dest_reg, Rc::clone(&label)),
                    //     format!("&{label}").into(),
                    // ),
                // };

                return Some(dest_reg.into());
            }
            unreachable!("can only reference identifiers")
        }

        let lhs_reg = self.expression(node.expr)?;

        match (lhs_type, node.op) {
            (Type::Int(0), PrefixOp::Neg) => {
                self.use_reg(lhs_reg, Size::Quad);
                let zero_reg = self.get_int_reg();
                self.use_reg(zero_reg.into(), Size::Quad);
                self.insert_with_comment(Instruction::Lghi(zero_reg, 0), format!("negate {lhs_reg}").into());
                let dest_reg = self.infix_helper(zero_reg.into(), lhs_reg, InfixOp::Minus, Type::Int(0));
                self.release_reg(lhs_reg);
                self.release_reg(zero_reg.into());
                Some(dest_reg)
            }
            (Type::Float(0), PrefixOp::Neg) => {
                todo!("implement this");
                // let dest_reg = self.get_float_reg();
                // self.insert(Instruction::FNeg(dest_reg, lhs_reg.into()));
                // Some(dest_reg.to_reg())
            }
            (Type::Int(0), PrefixOp::Not) => {
                todo!("implement this");
                // let dest_reg = self.get_int_reg();
                // self.insert(Instruction::Not(dest_reg, lhs_reg.into()));
                // Some(dest_reg.to_reg())
            }
            (Type::Bool(0), PrefixOp::Not) => {
                todo!("implement this");
                // let dest_reg = self.get_int_reg();
                // self.insert(Instruction::Seqz(dest_reg, lhs_reg.into()));
                // Some(dest_reg.to_reg())
            }
            (Type::Bool(1) | Type::Char(1), PrefixOp::Deref) => {
                todo!("implement this");
                // let dest_reg = self.get_int_reg();
                //
                // self.insert(Instruction::Lgr(dest_reg, lhs_reg.into()));
                //
                // self.insert_with_comment(
                //     Instruction::Lb(dest_reg, Pointer::Register(dest_reg, 0)),
                //     "deref".into(),
                // );
                //
                // Some(dest_reg.into())
            }
            (Type::Float(1), PrefixOp::Deref) => {
                todo!("implement this");
                // let dest_reg = self.get_float_reg();
                //
                // self.insert_with_comment(
                //     Instruction::Fld(dest_reg, Pointer::Register(lhs_reg.into(), 0)),
                //     "deref".into(),
                // );
                //
                // Some(dest_reg.into())
            }
            (Type::Int(_) | Type::Bool(_) | Type::Char(_) | Type::Float(_), PrefixOp::Deref) => {
                todo!("implement this");
                // let dest_reg = self.get_int_reg();
                //
                // self.insert(Instruction::Lgr(dest_reg, lhs_reg.into()));
                //
                // self.insert_with_comment(
                //     Instruction::Ld(dest_reg, Pointer::Register(dest_reg, 0)),
                //     "deref".into(),
                // );
                //
                // Some(dest_reg.into())
            }
            (t, o) => {
                unreachable!("other combinations cannot occur in prefix expressions: {t}: {o}")
            }
        }
    }

    /// Compiles an [`AnalyzedInfixExpr`].
    /// After compiling the lhs and rhs, the `infix_helper` is invoked.
    fn infix_expr(&mut self, node: AnalyzedInfixExpr<'tree>) -> Option<Register> {
        if node.lhs.result_type() == Type::Bool(0) && matches!(node.op, InfixOp::Or | InfixOp::And)
        {
        todo!("implement short circuiting.");
        //     let lhs = self.expression(node.lhs)?;
        //     let merge_block = self.gen_label("merge");
        //
        //     let (condition, comment) = match node.op == InfixOp::Or {
        //         true => (Condition::Ne, "||"),  // if the lhs is `true` ( || )
        //         false => (Condition::Eq, "&&"), // if the lhs is `false` ( && )
        //     };
        //
        //     // jump to the merge block if the result is determined by the lhs
        //     self.insert_with_comment(
        //         Instruction::BrCond(
        //             condition,
        //             lhs.into(),
        //             IntRegister::Zero,
        //             Rc::clone(&merge_block),
        //         ),
        //         comment.into(),
        //     );
        //
        //     // rhs is unused on release builds
        //     let _rhs = self.expression(node.rhs);
        //
        //     #[cfg(debug_assertions)]
        //     if let Some(rhs) = _rhs {
        //         assert_eq!(lhs, rhs);
        //     }
        //
        //     self.insert(Instruction::Jmp(Rc::clone(&merge_block)));
        //     self.blocks.push(Block::new(Rc::clone(&merge_block)));
        //     self.insert_at(&merge_block);
        //
        //     return Some(lhs)
        }
        match (node.lhs, node.rhs, node.op) {
            (AnalyzedExpression::Int(value), expr, InfixOp::Plus)
            | (expr, AnalyzedExpression::Int(value), InfixOp::Plus) => {
                let rhs_reg = self.expression(expr)?;
                self.insert(Instruction::Aghi(rhs_reg.into(), value.try_into().expect("offsett too large")));
                Some(rhs_reg)
            }
            (AnalyzedExpression::Ident(_), AnalyzedExpression::Int(0), InfixOp::Mul)
            | (AnalyzedExpression::Int(0), AnalyzedExpression::Int(_), InfixOp::Mul) => {
                let res_reg = self.get_int_reg();
                self.insert(Instruction::Lghi(res_reg, 0));
                Some(res_reg.to_reg())
            }
            (AnalyzedExpression::Int(0), expr, InfixOp::Mul)
            | (expr, AnalyzedExpression::Int(0), InfixOp::Mul) => {
                let res_reg = self
                    .expression(expr)
                    .expect("operand is always int register");
                self.insert(Instruction::Lghi(res_reg.into(), 0));
                Some(res_reg)
            }
            (lhs, rhs, op) => {
                let lhs_type = lhs.result_type();

                // mark LHS register as used and compile RHS
                let lhs_reg = self.expression(lhs)?;
                self.use_reg(lhs_reg, Size::from(lhs_type));

                let rhs_reg = self.expression(rhs)?;

                // set LHS register as unused
                self.release_reg(lhs_reg);

                let res = self.infix_helper(lhs_reg, rhs_reg, op, lhs_type);

                Some(res)
            }
        }
    }

    fn int_division_helper(
        &mut self,
        lhs: IntRegister,
        rhs: IntRegister,
        dest_reg: IntRegister,
        output: DivisionOutput,
    ) {
                // We are using the even register pair specified below.
                const EVEN_PAIR_LOW: IntRegister = IntRegister::R2;
                const EVEN_PAIR_HIGH: IntRegister = IntRegister::R4;

                // The results of the division will be placed in the odd register pair specified
                // below.
                const ODD_PAIR_LOW: IntRegister = IntRegister::R1;
                const ODD_PAIR_HIGH: IntRegister = IntRegister::R3;

                const REGS_USED_FOR_DIVISION: [IntRegister; 4] = [
                    EVEN_PAIR_LOW,
                    EVEN_PAIR_HIGH,
                    ODD_PAIR_LOW,
                    ODD_PAIR_HIGH,
                ];

                let mut saved = vec![];

                for r in REGS_USED_FOR_DIVISION {
                    if self.reg_in_use(&(r.to_reg())) {
                        let offset = self.save_ireg_on_stack(r, Some("save before division".into()));
                        saved.push((r, offset));
                    }
                }

                // Move the lhs, rhs into the even register pair (input pair).
                if lhs != ODD_PAIR_HIGH {
                    self.insert_movi(ODD_PAIR_HIGH, lhs);
                }

                if rhs != EVEN_PAIR_HIGH {
                    self.insert_movi(EVEN_PAIR_HIGH, rhs);
                }

                self.insert(Instruction::Div64(EVEN_PAIR_LOW, EVEN_PAIR_HIGH));

                // Extract the quotient / remainder from the odd-numbered register pair.
                match (output, dest_reg) {
                    (DivisionOutput::Quotient, ODD_PAIR_HIGH) => {},
                    (DivisionOutput::Quotient, _) => {
                        self.insert_movi(dest_reg, ODD_PAIR_HIGH);
                    },
                    (DivisionOutput::Remainder, EVEN_PAIR_LOW) => {},
                    (DivisionOutput::Remainder, _) => {
                        self.insert_movi(dest_reg, EVEN_PAIR_LOW);
                    },
                }

                // Restore saved registers.
                for (r, offset) in saved {
                    self.restore_ireg_from_stack(r, offset)
                }
    }

    fn float_division_helper(
        &mut self,
        lhs: FloatRegister,
        rhs: FloatRegister,
        dest_reg: FloatRegister,
        output: DivisionOutput,
    ) {
                // We are using the even register pair specified below.
                const EVEN_PAIR_LOW: FloatRegister = FloatRegister::F2;
                const EVEN_PAIR_HIGH: FloatRegister = FloatRegister::F4;

                // The results of the division will be placed in the odd register pair specified
                // below.
                const ODD_PAIR_LOW: FloatRegister = FloatRegister::F1;
                const ODD_PAIR_HIGH: FloatRegister = FloatRegister::F3;

                const REGS_USED_FOR_DIVISION: [FloatRegister; 4] = [
                    EVEN_PAIR_LOW,
                    EVEN_PAIR_HIGH,
                    ODD_PAIR_LOW,
                    ODD_PAIR_HIGH,
                ];

                let mut saved = vec![];

                for r in REGS_USED_FOR_DIVISION {
                    if self.reg_in_use(&(r.to_reg())) {
                        let offset = self.save_freg_on_stack(r, Some("save before division (f)".into()));
                        saved.push((r, offset));
                    }
                }

                // Move the lhs, rhs into the even register pair (input pair).
                if lhs != EVEN_PAIR_LOW {
                    self.insert_movf(EVEN_PAIR_LOW, lhs);
                }

                if rhs != EVEN_PAIR_HIGH {
                    self.insert_movf(EVEN_PAIR_HIGH, rhs);
                }

                self.insert(Instruction::Ddbr(EVEN_PAIR_LOW, EVEN_PAIR_HIGH));

                // Extract the quotient / remainder from the odd-numbered register pair.
                match (output, dest_reg) {
                    (DivisionOutput::Quotient, EVEN_PAIR_LOW) => {},
                    (DivisionOutput::Quotient, _) => {
                        self.insert_movf(dest_reg, EVEN_PAIR_LOW);
                    },
                    (DivisionOutput::Remainder, EVEN_PAIR_LOW) => {},
                    (DivisionOutput::Remainder, _) => {
                        self.insert_movf(dest_reg, EVEN_PAIR_LOW);
                    },
                }

                // Restore saved registers.
                for (r, offset) in saved {
                    if r == dest_reg {
                        panic!("TODO")
                    }

                    self.restore_freg_from_stack(r, offset)
                }
    }

    /// Helper function which handles parts of infix expressions.
    fn infix_helper(&mut self, lhs: Register, rhs: Register, op: InfixOp, type_: Type) -> Register {
        // creates the two result registers
        // eventually, just one of the two is used
        let mut dest_regi = self.get_int_reg();
        let dest_regf = self.get_float_reg();

        match (type_, op) {
            (Type::Int(0), InfixOp::Plus) => {
                self.insert(Instruction::Add64(lhs.into(), rhs.into()));
                self.insert_movi(dest_regi, lhs.into());
                dest_regi.into()
            }
            (Type::Int(0), InfixOp::Minus) => {
                self.insert(Instruction::Sub64(lhs.into(), rhs.into()));
                self.insert_movi(dest_regi, lhs.into());
                dest_regi.into()
            }
            (Type::Char(0), InfixOp::Plus) => {
                todo!("implement this");
                // self.insert(Instruction::Add(dest_regi, lhs.into(), rhs.into()));

                // TODO: is this really required?
                // self.use_reg(dest_regi.into(), Size::Byte);
                // let mask = self.get_int_reg();
                // self.insert(Instruction::Lghi(mask, 0x7f));
                // self.release_reg(dest_regi.into());
                //
                // self.insert(Instruction::And(dest_regi, dest_regi, mask));

                // dest_regi.into()
            }
            (Type::Char(0), InfixOp::Minus) => {
                todo!("implement this");
                // self.insert(Instruction::Sub(dest_regi, lhs.into(), rhs.into()));
                //
                // // TODO: is this really required?
                // self.use_reg(dest_regi.into(), Size::Byte);
                // let mask = self.get_int_reg();
                // self.insert(Instruction::Lghi(mask, 0x7f));
                // self.release_reg(dest_regi.into());
                //
                // self.insert(Instruction::And(dest_regi, dest_regi, mask));
                //
                // dest_regi.into()
            }
            // TODO
            (Type::Int(0), InfixOp::Mul) => {
                let lhs_int: IntRegister = lhs.into();

                self.insert(Instruction::Mul64(lhs.into(), rhs.into()));

                if lhs_int != dest_regi {
                    self.insert_movi(dest_regi, lhs.into());
                }

                dest_regi.into()
            }
            (Type::Int(0), InfixOp::Div) => {
                self.int_division_helper(
                    lhs.into(),
                    rhs.into(),
                    dest_regi,
                    DivisionOutput::Quotient,
                );

                dest_regi.into()
            }
            (Type::Int(0), InfixOp::Rem) => {
                self.int_division_helper(
                    lhs.into(),
                    rhs.into(),
                    dest_regi,
                    DivisionOutput::Remainder,
                );

                dest_regi.into()
            }
            (Type::Int(0), InfixOp::Pow) => {
                self.use_reg(lhs, Size::Quad);
                self.use_reg(rhs, Size::Quad);
                self.use_reg(dest_regi.into(), Size::Quad);

                dbg!(lhs, rhs, dest_regi);

                let dest = self
                .__rush_internal_pow_int(lhs.into(), rhs.into())
                .to_reg();

                self.use_reg(lhs, Size::Quad);
                self.use_reg(rhs, Size::Quad);
                self.use_reg(dest_regi.into(), Size::Quad);

                if dest != dest_regi.into() {
                    self.insert_with_comment(
                        Instruction::Lgr(dest_regi.into(), dest),
                        "result of pow_int".into(),
                    );
                }

                dest_regi.into()
            },
            // (Type::Int(0), InfixOp::Shl) => {
            //     self.insert(Instruction::Sll(dest_regi, lhs.into(), rhs.into()));
            //     dest_regi.into()
            // }
            // (Type::Int(0), InfixOp::Shr) => {
            //     self.insert(Instruction::Sra(dest_regi, lhs.into(), rhs.into()));
            //     dest_regi.into()
            // }
            // (Type::Int(0) | Type::Bool(0), InfixOp::BitOr | InfixOp::Or) => {
            //     self.insert(Instruction::Or(dest_regi, lhs.into(), rhs.into()));
            //     dest_regi.into()
            // }
            // (Type::Int(0) | Type::Bool(0), InfixOp::BitAnd | InfixOp::And) => {
            //     self.insert(Instruction::And(dest_regi, lhs.into(), rhs.into()));
            //     dest_regi.into()
            // }
            // (Type::Int(0) | Type::Bool(0), InfixOp::BitXor) => {
            //     self.insert(Instruction::Xor(dest_regi, lhs.into(), rhs.into()));
            //     dest_regi.into()
            // }
            // even if not all ops are allowed for char and bool, the analyzer would not accept
            // illegal programs, therefore this is ok.
            (
                Type::Int(0) | Type::Char(0) | Type::Bool(0),
                op @ (    InfixOp::Eq
                | InfixOp::Neq
                | InfixOp::Lt
                | InfixOp::Lte
                | InfixOp::Gte ),
            ) => {
                self.insert(Instruction::Comment(format!("begin comparison {op}").into()));

                // Insert compare instruction.
                self.insert_with_comment(Instruction::CompareIReg(
                    lhs.into(),
                    rhs.into(),
                ), format!("compare ({lhs} {op} {rhs})").into());

                //
                // Now we have two possible paths: one which places `true` in `dest_regi`,
                // and the other which leaves `false` in the register.
                //

                let true_label = self.gen_label("comparison_true");
                let merge_label = self.gen_label("comparison_merge");

                let jump_instruction = match op {
                    InfixOp::Eq => Instruction::BranchEq(true_label.clone()),
                    InfixOp::Neq => Instruction::BranchNotEq(true_label.clone()),
                    InfixOp::Lt =>  Instruction::BranchLessThan(true_label.clone()),
                    InfixOp::Lte => Instruction::BranchNotGreaterThan(true_label.clone()),
                    InfixOp::Gte => Instruction::BranchNotLessThan(true_label.clone()),
                    _ => unreachable!("checked above"),
                };

                // Place jump instruction, which would skip the default `false`.
                self.insert(jump_instruction);

                // Place `false` as the default.
                self.insert_with_comment(Instruction::Lghi(dest_regi, false as i16), "`false` case of comp".into());
                self.insert(Instruction::Jmp(Rc::clone(&merge_label)));

                self.blocks.push(Block::new(Rc::clone(&true_label)));
                self.insert_at(&Rc::clone(&true_label));

                // Place `true` as the non-default.
                self.insert_with_comment(Instruction::Lghi(dest_regi, true as i16), "`true` case of comp".into());


                self.blocks.push(Block::new(Rc::clone(&merge_label)));
                self.insert_at(&Rc::clone(&merge_label));
                self.insert(Instruction::Comment(format!("end comparison {op}").into()));

                dest_regi.to_reg()
            }
            (
                // even if not all ops are allowed for char and bool, the analyzer would not accept
                // illegal programs, therefore this is ok.
                Type::Int(0) | Type::Char(0) | Type::Bool(0),
                op @ InfixOp::Gt,
            ) => {
                self.insert(Instruction::Comment(format!("begin comparison {op}").into()));

                // Insert compare instruction.
                self.insert_with_comment(Instruction::CompareIReg(
                    lhs.into(),
                    rhs.into(),
                ), format!("compare ({lhs} {op} {rhs})").into());

                // Store CC and program mask in register CMP_CC_SOURCE.
                const REGS_USED_FOR_CMP: [IntRegister; 2] =[ IntRegister::R0, IntRegister::R1];
                const CMP_CC_SOURCE: IntRegister = REGS_USED_FOR_CMP[0];
                const CMP_CC_BASE_ADDR: IntRegister = REGS_USED_FOR_CMP[1];

                let mut saved = vec![];

                for r in REGS_USED_FOR_CMP {
                    if self.reg_in_use(&(r.to_reg())) {
                        let offset = self.save_ireg_on_stack(r, Some("save before compare".into()));
                        saved.push((r, offset));
                    }
                }

                self.insert_with_comment(
                    Instruction::InsertProgramMask(REGS_USED_FOR_CMP[0]),
                    format!("store CC in {CMP_CC_SOURCE}").into(),
                );


                //
                // Shift so that the low 2 bits are the CC.
                //
                // Place shift amount in dest_regi
                let shift_amount = 29;
                self.insert_with_comment(Instruction::Lghi(CMP_CC_BASE_ADDR     , shift_amount), "CC shift amount".into());
                self.insert_with_comment(
                    Instruction::ShiftRightSingle(dest_regi, CMP_CC_SOURCE, 0, CMP_CC_BASE_ADDR),
                    "CC in low 2 bits".into(),
                );

                self.insert(Instruction::Comment(format!("end comparison {op}").into()));

                // Restore saved registers.
                for (r, offset) in saved {
                    if r == dest_regi {
                        let temp = self.get_int_reg();
                        self.insert_movi(temp, dest_regi);
                        dest_regi = temp;
                    }

                    self.restore_ireg_from_stack(r, offset)
                }

                dest_regi.to_reg()
            }
            // (
            //     Type::Float(0),
            //     InfixOp::Eq
            //     | InfixOp::Neq
            //     | InfixOp::Lt
            //     | InfixOp::Lte
            //     | InfixOp::Gt
            //     | InfixOp::Gte,
            // ) => {
            //     self.insert(Instruction::SetFloatCondition(
            //         Condition::from(op),
            //         dest_regi,
            //         lhs.into(),
            //         rhs.into(),
            //     ));
            //     dest_regi.into()
            // }
            (Type::Float(0), InfixOp::Plus) => {
                let lhs_float: FloatRegister = lhs.into();

                self.insert(Instruction::Adbr(lhs_float, rhs.into()));

                if lhs_float != dest_regf {
                    self.insert_movf(dest_regf, lhs_float);
                }

                dest_regf.into()
            }
            (Type::Float(0), InfixOp::Minus) => {
                let lhs_float: FloatRegister = lhs.into();

                self.insert(Instruction::Sdbr(lhs_float, rhs.into()));

                if lhs_float != dest_regf {
                    self.insert_movf(dest_regf, lhs_float);
                }

                dest_regf.into()
            }
            (Type::Float(0), InfixOp::Mul) => {
                let lhs_float: FloatRegister = lhs.into();

                self.insert(Instruction::Mdbr(lhs.into(), rhs.into()));

                if lhs_float != dest_regf {
                    self.insert_movf(dest_regf, lhs.into());
                }

                dest_regf.into()
            }
            (Type::Float(0), InfixOp::Div) => {
                self.float_division_helper(
                    lhs.into(),
                    rhs.into(),
                    dest_regf,
                    DivisionOutput::Quotient,
                );

                dest_regf.into()
            }
            (t, o) => unreachable!("the analyzer does not allow other combinations: {t}: {o}"),
        }
    }

    /// Compiles an [`AnalyzedAssignExpr`].
    /// Performs simple assignments and complex operator-backed assignments.
    /// For the latter, the assignee's current value is loaded into a temporary register.
    /// Following that, the operation is performed by `self.infix_helper`.
    /// Lastly, a correct store instruction is used to assign the resulting value to the assignee.
    fn assign_expr(&mut self, node: AnalyzedAssignExpr<'tree>) {
        let rhs_type = node.expr.result_type();

        // let (mut src_ptr, mut assignee_type) = self.load_variable_from_name(node.assignee).clone();
        let assignee_var = self.resolve_variable(node.assignee);

        let ptr_reg = self.get_int_reg();

        // if the lhs is an indirected pointer, perform required indirections
        if node.assignee_ptr_count > 0 {
            todo!("pointers");
            // TODO: pointers
            // let mut ptr_count = node.assignee_ptr_count;
            // // TODO: is this really required?
            // self.use_reg(ptr_reg.into(), Size::Quad);
            //
            // while ptr_count > 0 {
            //     self.insert_with_comment(
            //         Instruction::Load64(
            //             ptr_reg,
            //             src_ptr.clone().expect("analyzer guarantees valid pointers")
            //         ),
            //         "deref".into(),
            //     );
            //     src_ptr = Some(IntRegisterPointer(ptr_reg, 0));
            //     assignee_type = assignee_type
            //         .sub_deref()
            //         .expect("the analyzer guarantees valid usage of pointers");
            //     ptr_count -= 1;
            // }
        }

        // holds the value of the rhs (either simple or the result of an operation)
        'outer: {
            let rhs_reg = match node.op {
                AssignOp::Basic => match self.expression(node.expr) {
                    Some(reg) => reg,
                    None => return,
                },
                AssignOp::Pow => {
                    // Load actual assignee address.
                    let assignee_ptr = self.load_variable_from_name(node.assignee).unwrap();

                    // load value from the lhs
                    let lhs = self
                        // `clone` only clones a [`Rc`]
                        .load_value_from_pointer(
                            assignee_ptr,
                            assignee_var.type_,
                            node.assignee,
                        );
                    self.use_reg(lhs, Size::from(assignee_var.type_));

                    // compile the rhs
                    let Some(rhs) = self.expression(node.expr) else { break 'outer };
                    self.use_reg(rhs, Size::from(rhs_type));

                    // call the `pow` corelib function using the `infix_helper`
                    let res = self.infix_helper(lhs, rhs, InfixOp::from(node.op), assignee_var.type_);

                    self.release_reg(lhs);
                    self.release_reg(rhs);
                    res
                }
                _ => {
                    // compile the rhs
                    let Some(rhs) = self.expression(node.expr) else { break 'outer};
                    self.use_reg(rhs, Size::from(rhs_type));

                    // Load actual assignee address.
                    let assignee_ptr = self.load_variable_from_name(node.assignee).unwrap();

                    // load value from the lhs
                    let lhs = self
                        // `clone` only clones a [`Rc`]
                        .load_value_from_pointer(
                            assignee_ptr.clone(),
                            assignee_var.type_,
                            node.assignee,
                        );
                    self.use_reg(lhs, Size::from(assignee_var.type_));

                    // perform pre-assign operation using the infix helper
                    let res = self.infix_helper(lhs, rhs, InfixOp::from(node.op), assignee_var.type_);

                    self.release_reg(lhs);
                    self.release_reg(rhs);
                    res
                }
            };

            // Load actual assignee address.
            let assignee_ptr = self.load_variable_from_name(node.assignee);

            if let Some(ptr) = assignee_ptr {
                match rhs_type {
                    Type::Float(0) => self.insert(Instruction::StoreGeneric(rhs_reg, ptr)),
                    Type::Bool(0) | Type::Char(0) => {
                        // Clear the register completely beforehand.
                        // TODO: implement this.
                        self.insert(Instruction::Store8(rhs_reg.into(), ptr))
                    }
                    Type::Int(_) | Type::Bool(_) | Type::Char(_) | Type::Float(_) => {
                        dbg!(&ptr);
                        self.insert(Instruction::Store64(rhs_reg.into(), ptr))
                    }
                    Type::Unit | Type::Never => {} // ignore these types
                    Type::Unknown => unreachable!("the analyzer would have failed"),
                    _ => todo!("float, bool not implemented")
                }
            }
        }

        // release the ptr register if it was used
        if node.assignee_ptr_count > 0 {
            self.release_reg(ptr_reg.into());
        }
    }

    /// Compiles an [`AnalyzedCastExpr`].
    /// When casting to `char` values, cast functions from the `corelib` are invoked.
    fn cast_expr(&mut self, node: AnalyzedCastExpr<'tree>) -> Option<Register> {
        let lhs_type = node.expr.result_type();
        let lhs_reg = self.expression(node.expr)?;

        // block the use of the lhs temporarily
        self.use_reg(lhs_reg, Size::from(lhs_type));

        let res = match (lhs_type, node.type_) {
            // nop: just return the lhs
            (lhs, rhs) if lhs == rhs => lhs_reg,
            (Type::Bool(0), Type::Int(0))
            | (Type::Bool(0), Type::Char(0))
            | (Type::Char(0), Type::Int(0)) => lhs_reg,
            // integer base type casts
            (Type::Int(0), Type::Float(0)) => {
                let dest_reg = self.get_float_reg();
                self.insert(Instruction::ConvertFromFixed(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            // (Type::Char(0) | Type::Bool(0), Type::Float(0)) => {
            //     let dest_reg = self.get_float_reg();
            //     self.insert(Instruction::CastByteToFloat(dest_reg, lhs_reg.into()));
            //     dest_reg.to_reg()
            // }
            // (Type::Int(0) | Type::Char(0), Type::Bool(0)) => {
            //     let dest_reg = self.get_int_reg();
            //     self.insert(Instruction::Snez(dest_reg, lhs_reg.into()));
            //     dest_reg.to_reg()
            // }
            // (Type::Int(0), Type::Char(0)) => self
            //     .__rush_internal_cast_int_to_char(lhs_reg.into())
            //     .to_reg(),
            // // float base type casts
            (Type::Float(0), Type::Int(0)) => {
                const FLOAT_TO_INT_ROUNDING_MODE:u8 =5;

                let dest_reg = self.get_int_reg();
                self.insert(Instruction::ConvertToFixed(dest_reg, FLOAT_TO_INT_ROUNDING_MODE, lhs_reg.into()));

                dest_reg.into()
            }
            // (Type::Float(0), Type::Char(0)) => self
            //     .__rush_internal_cast_float_to_char(lhs_reg.into())
            //     .to_reg(),
            // (Type::Float(0), Type::Bool(0)) => {
            //     // get a `.rodata` label which holds a float zero to compare to
            //     let float_zero_label = match self
            //         .rodata_section
            //         .iter()
            //         .find(|o| o.data == DataObjType::Float(0.0))
            //     {
            //         Some(obj) => Rc::clone(&obj.label),
            //         None => {
            //             // create a float constant with the value 0
            //             let label = format!("float_constant_{}", self.rodata_section.len()).into();
            //             self.rodata_section.push(DataObj {
            //                 label: Rc::clone(&label),
            //                 data: DataObjType::Float(0.0),
            //             });
            //             label
            //         }
            //     };
            //
            //     // load value from float constant into a free float register
            //     let zero_float_reg = self.get_float_reg();
            //     self.insert(Instruction::Fld(
            //         zero_float_reg,
            //         Pointer::Label(float_zero_label),
            //     ));
            //
            //     // compare the float to `0.0`
            //     let dest_reg = self.get_int_reg();
            //     self.insert(Instruction::SetFloatCondition(
            //         Condition::Ne,
            //         dest_reg,
            //         zero_float_reg,
            //         lhs_reg.into(),
            //     ));
            //
            //     // return the result of the comparison
            //     dest_reg.to_reg()
            // }
            // TODO: implement the rest.
            _ => unreachable!("cannot use other combinations in a typecast"),
        };

        // release the block of the lhs
        self.release_reg(lhs_reg);

        Some(res)
    }

    /// Compiles an [`AnalyzedIfExpr`].
    /// The result of the expression is saved in a single register (corresponding to the result type).
    fn if_expr(&mut self, node: AnalyzedIfExpr<'tree>) -> Option<Register> {
        // (bool) result of the condition
        let cond_reg = self.expression(node.clone().cond)?;

        // will later hold the result of the branch
        let res_reg = match node.result_type {
            Type::Float(0) => Some(self.get_float_reg().to_reg()),
            Type::Int(0) | Type::Bool(0) | Type::Char(0) => Some(self.get_int_reg().to_reg()),
            _ => None, // other types require no register
        };

        let merge_block = self.gen_label("merge");
        let else_block_label = self.gen_label("else");

        // if the condition evaluated to `false`, jump to the `else` or `merge` block
        self.insert(Instruction::CompareIntImm(cond_reg.into(), false as i8));
        self.insert(Instruction::BranchEq(
            if node.else_block.is_some() {
                Rc::clone(&else_block_label)
            } else {
                Rc::clone(&merge_block)
            }
        ));

        // self.insert(Instruction::BrCond(
        //     Condition::Eq,
        //     cond_reg.into(),
        //     IntRegister::Zero,
        //     if node.else_block.is_some() {
        //         Rc::clone(&else_block_label)
        //     } else {
        //         Rc::clone(&merge_block)
        //     },
        // ));

        let then_reg = self.block(node.clone().then_block);

        // if the `then` block returns a register other than res, move the block register into res
        match (res_reg, then_reg) {
            (Some(Register::Int(res)), Some(Register::Int(then_reg))) => {
                self.insert_movi(res, then_reg);
            }
            (Some(Register::Float(res)), Some(Register::Float(then_reg))) => {
                self.insert_movf(res, then_reg);
            }
            _ => {}
        }

        // jump to the `merge` block
        self.insert_jmp(Rc::clone(&merge_block), None);

        // if there is an `else` block, compile it
        if let Some(else_block) = node.else_block {
            self.blocks.push(Block::new(Rc::clone(&else_block_label)));
            self.insert_at(&else_block_label);
            let else_reg = self.block(else_block);

            // if the block returns a register other than res, move it into `res_reg`
            match (res_reg, else_reg) {
                (Some(Register::Int(res)), Some(Register::Int(else_reg))) => {
                    self.insert_movi(res, else_reg);
                }
                (Some(Register::Float(res)), Some(Register::Float(else_reg))) => {
                    self.insert_movf(res, else_reg);
                }
                _ => {}
            }

            // jumps to `merge` from the `else` block
            self.insert_jmp(Rc::clone(&merge_block), None);
        }

        // set the cursor position to the end of the `merge` block
        self.blocks.push(Block::new(Rc::clone(&merge_block)));
        self.insert_at(&merge_block);

        res_reg
    }
}

impl<'tree> Default for Compiler<'tree> {
    fn default() -> Self {
        Self::new()
    }
}

use std::{borrow::Cow, collections::HashMap, rc::Rc};

use rush_analyzer::{ast::*, AssignOp, InfixOp, PrefixOp, Type};

use crate::{
    instruction::{Block, CommentConfig, Condition, Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register},
    utils::{DataObj, DataObjType, Function, Loop, Size, Variable},
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

    /// Compiles the source AST into a RISC-V targeted Assembly program.
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

        // generate Assembly
        self.codegen(comment_config)
    }

    /// Generates the Assembly representation from the compiled program.
    /// This function is invoked in the last step of compilation and only generates the output.
    fn codegen(&self, comment_config: &CommentConfig) -> String {
        let mut output = String::new();

        // `.global` label exports
        output += &self
            .exports
            .iter()
            .map(|e| format!(".global {e}\n"))
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
                "\n.section .rodata\n{}",
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
        self.insert(Instruction::Call(main_label.into()));
        // exit with code 0 by default
        self.insert(Instruction::Li(IntRegister::A0, 0));
        self.insert(Instruction::Call("exit".into()));

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
            (Type::Int(0), AnalyzedExpression::Int(val)) => DataObjType::Dword(val),
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
        let mut int_cnt = 0; // 0 = a0
        let mut float_cnt = 0; // 0 = fa0

        // specifies the memory offset to use when params are spilled
        // is incremented in steps of 8
        let mut mem_offset = 0;

        // save all param values in the current scope / on the stack
        for param in &node.params {
            match param.type_ {
                Type::Float(0) => {
                    match FloatRegister::nth_param(float_cnt) {
                        Some(reg) => {
                            let offset = self.get_offset(Size::Dword);

                            param_store_instructions.push((
                                Instruction::Fsd(reg, Pointer::Register(IntRegister::Fp, offset)),
                                Some(format!("param {} = {reg}", param.name).into()),
                            ));

                            // insert the param into the scope
                            self.scope_mut().insert(
                                param.name,
                                Variable {
                                    type_: param.type_,
                                    value: Some(Pointer::Register(IntRegister::Fp, offset)),
                                },
                            );
                        }
                        None => {
                            // if there are spilled params, insert their location into the scope
                            self.scope_mut().insert(
                                param.name,
                                Variable {
                                    type_: param.type_,
                                    value: Some(Pointer::Register(IntRegister::Fp, mem_offset)),
                                },
                            );
                            mem_offset += 8;
                        }
                    }
                    float_cnt += 1;
                }
                Type::Int(_) | Type::Char(_) | Type::Bool(_) | Type::Float(_) => {
                    match IntRegister::nth_param(int_cnt) {
                        Some(reg) => {
                            let size = Size::from(param.type_);
                            let offset = self.get_offset(size);

                            // use `sb` or `sd` depending on the size
                            match size {
                                Size::Byte => param_store_instructions.push((
                                    Instruction::Sb(
                                        reg,
                                        Pointer::Register(IntRegister::Fp, offset),
                                    ),
                                    Some(format!("param {} = {reg}", param.name).into()),
                                )),
                                Size::Dword => param_store_instructions.push((
                                    Instruction::Sd(
                                        reg,
                                        Pointer::Register(IntRegister::Fp, offset),
                                    ),
                                    Some(format!("param {} = {reg}", param.name).into()),
                                )),
                            }

                            // insert the param into the scope
                            self.scope_mut().insert(
                                param.name,
                                Variable {
                                    type_: param.type_,
                                    value: Some(Pointer::Register(IntRegister::Fp, offset)),
                                },
                            );
                        }
                        None => {
                            // if there are spilled params, insert their location into the scope
                            self.scope_mut().insert(
                                param.name,
                                Variable {
                                    type_: param.type_,
                                    value: Some(Pointer::Register(IntRegister::Fp, mem_offset)),
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
        // for `int`, `bool`, and `char`:   `a0`
        // for `float`:                     `fa0`
        if let Some(expr) = node.expr {
            // if the result register does not match the desired register, insert a move instruction
            match self.expression(expr) {
                Some(Register::Int(reg)) => {
                    self.insert(Instruction::Mv(IntRegister::A0, reg));
                }
                Some(Register::Float(reg)) => {
                    self.insert(Instruction::Fmv(FloatRegister::Fa0, reg));
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
                Some(Register::Int(IntRegister::A0)) => {}      // already in correct register
                Some(Register::Float(FloatRegister::Fa0)) => {} // already in correct register
                Some(Register::Int(reg)) => self.insert(Instruction::Mv(IntRegister::A0, reg)),
                Some(Register::Float(reg)) => {
                    self.insert(Instruction::Fmv(FloatRegister::Fa0, reg))
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
        self.insert(Instruction::Beqz(cond.into(), Rc::clone(&after_loop)));

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
            .insert(node.ident, Variable { type_, value: ptr });

        //// CONDITION ////
        self.insert_at(&for_head);

        self.insert(Instruction::Comment("for condition".into()));

        // if the cond is `!`, return here
        let Some(cond) = self.expression(node.cond) else {
            return
        };

        self.insert(Instruction::Beqz(cond.into(), Rc::clone(&after_loop_label)));

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
            .insert(node.name, Variable { type_, value: ptr });
    }

    fn save_expr_on_stack(
        &mut self,
        node: AnalyzedExpression<'tree>,
        comment_prefix: String,
    ) -> Option<Pointer> {
        let type_ = node.result_type();
        let reg = self.expression(node)?;
        let comment = format!("{comment_prefix} = {reg}").into();
        let offset = self.get_offset(Size::from(type_));

        match reg {
            Register::Int(reg) => match type_ {
                Type::Bool(0) | Type::Char(0) => self.insert_with_comment(
                    Instruction::Sb(reg, Pointer::Register(IntRegister::Fp, offset)),
                    comment,
                ),
                Type::Int(_) | Type::Float(_) | Type::Bool(_) | Type::Char(_) => self
                    .insert_with_comment(
                        Instruction::Sd(reg, Pointer::Register(IntRegister::Fp, offset)),
                        comment,
                    ),
                _ => unreachable!("only the types above use int registers"),
            },
            Register::Float(reg) => self.insert_with_comment(
                Instruction::Fsd(reg, Pointer::Register(IntRegister::Fp, offset)),
                comment,
            ),
        };

        Some(Pointer::Register(IntRegister::Fp, offset))
    }

    /// Compiles an [`AnalyzedExpression`].
    pub(crate) fn expression(&mut self, node: AnalyzedExpression<'tree>) -> Option<Register> {
        match node {
            AnalyzedExpression::Int(value) => {
                let dest_reg = self.get_int_reg();
                self.insert(Instruction::Li(dest_reg, value));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Bool(value) => {
                let dest_reg = self.get_int_reg();
                self.insert(Instruction::Li(dest_reg, value as i64));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Char(value) => {
                let dest_reg = self.get_int_reg();
                self.insert(Instruction::Li(dest_reg, value as i64));
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

                // load the value from the data label into `dest_reg`
                self.insert(Instruction::Fld(
                    dest_reg,
                    Pointer::Label(float_value_label),
                ));

                Some(Register::Float(dest_reg))
            }
            AnalyzedExpression::Ident(ident) => {
                // if this is a placeholder or dummy variable, return `None`
                let var = match self.resolve_name(ident.ident) {
                    Variable { value: None, .. } => return None,
                    var => var,
                };
                // `clone` is okay here, since it only clones a `Rc`
                Some(self.load_value_from_pointer(var.value.clone()?, var.type_, ident.ident))
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
                let var = self.resolve_name(ident.ident);
                let dest_reg = self.get_int_reg();

                match var
                    .value
                    .clone()
                    .expect("the analyzer guarantees valid pointers")
                {
                    Pointer::Register(_, offset) => {
                        self.insert_with_comment(
                            Instruction::Addi(dest_reg, IntRegister::Fp, offset),
                            format!("&{}", ident.ident).into(),
                        );
                    }
                    Pointer::Label(label) => self.insert_with_comment(
                        Instruction::La(dest_reg, Rc::clone(&label)),
                        format!("&{label}").into(),
                    ),
                };

                return Some(dest_reg.into());
            }
            unreachable!("can only reference identifiers")
        }

        let lhs_reg = self.expression(node.expr)?;

        match (lhs_type, node.op) {
            (Type::Int(0), PrefixOp::Neg) => {
                let dest_reg = self.get_int_reg();
                self.insert(Instruction::Neg(dest_reg, lhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Float(0), PrefixOp::Neg) => {
                let dest_reg = self.get_float_reg();
                self.insert(Instruction::FNeg(dest_reg, lhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Int(0), PrefixOp::Not) => {
                let dest_reg = self.get_int_reg();
                self.insert(Instruction::Not(dest_reg, lhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Bool(0), PrefixOp::Not) => {
                let dest_reg = self.get_int_reg();
                self.insert(Instruction::Seqz(dest_reg, lhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Bool(1) | Type::Char(1), PrefixOp::Deref) => {
                let dest_reg = self.get_int_reg();

                self.insert(Instruction::Mv(dest_reg, lhs_reg.into()));

                self.insert_with_comment(
                    Instruction::Lb(dest_reg, Pointer::Register(dest_reg, 0)),
                    "deref".into(),
                );

                Some(dest_reg.into())
            }
            (Type::Float(1), PrefixOp::Deref) => {
                let dest_reg = self.get_float_reg();

                self.insert_with_comment(
                    Instruction::Fld(dest_reg, Pointer::Register(lhs_reg.into(), 0)),
                    "deref".into(),
                );

                Some(dest_reg.into())
            }
            (Type::Int(_) | Type::Bool(_) | Type::Char(_) | Type::Float(_), PrefixOp::Deref) => {
                let dest_reg = self.get_int_reg();

                self.insert(Instruction::Mv(dest_reg, lhs_reg.into()));

                self.insert_with_comment(
                    Instruction::Ld(dest_reg, Pointer::Register(dest_reg, 0)),
                    "deref".into(),
                );

                Some(dest_reg.into())
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
            let lhs = self.expression(node.lhs)?;
            let merge_block = self.gen_label("merge");

            let (condition, comment) = match node.op == InfixOp::Or {
                true => (Condition::Ne, "||"),  // if the lhs is `true` ( || )
                false => (Condition::Eq, "&&"), // if the lhs is `false` ( && )
            };

            // jump to the merge block if the result is determined by the lhs
            self.insert_with_comment(
                Instruction::BrCond(
                    condition,
                    lhs.into(),
                    IntRegister::Zero,
                    Rc::clone(&merge_block),
                ),
                comment.into(),
            );

            // rhs is unused on release builds
            let _rhs = self.expression(node.rhs);

            #[cfg(debug_assertions)]
            if let Some(rhs) = _rhs {
                assert_eq!(lhs, rhs);
            }

            self.insert(Instruction::Jmp(Rc::clone(&merge_block)));
            self.blocks.push(Block::new(Rc::clone(&merge_block)));
            self.insert_at(&merge_block);

            return Some(lhs)
        }
        match (node.lhs, node.rhs, node.op) {
            (AnalyzedExpression::Int(value), expr, InfixOp::Plus)
            | (expr, AnalyzedExpression::Int(value), InfixOp::Plus) => {
                let rhs_reg = self.expression(expr)?;
                self.insert(Instruction::Addi(rhs_reg.into(), rhs_reg.into(), value));
                Some(rhs_reg)
            }
            (AnalyzedExpression::Ident(_), AnalyzedExpression::Int(0), InfixOp::Mul)
            | (AnalyzedExpression::Int(0), AnalyzedExpression::Int(_), InfixOp::Mul) => {
                let res_reg = self.get_int_reg();
                self.insert(Instruction::Li(res_reg, 0));
                Some(res_reg.to_reg())
            }
            (AnalyzedExpression::Int(0), expr, InfixOp::Mul)
            | (expr, AnalyzedExpression::Int(0), InfixOp::Mul) => {
                let res_reg = self
                    .expression(expr)
                    .expect("operand is always int register");
                self.insert(Instruction::Li(res_reg.into(), 0));
                Some(res_reg)
            }
            (AnalyzedExpression::Int(1), expr, InfixOp::Mul)
            | (expr, AnalyzedExpression::Int(1), InfixOp::Mul) => self.expression(expr),
            (expr, AnalyzedExpression::Int(value), InfixOp::Div) if value.count_ones() == 1 =>
            // checks if the divisor is a power of 2
            {
                let rhs_reg = self.expression(expr)?;
                self.insert(Instruction::Srai(
                    rhs_reg.into(),
                    rhs_reg.into(),
                    (value - 1).count_ones() as i64,
                ));
                Some(rhs_reg)
            }
            (AnalyzedExpression::Int(value), expr, InfixOp::Mul)
            | (expr, AnalyzedExpression::Int(value), InfixOp::Mul)
                if value.count_ones() == 1 =>
            // checks if the factor is a power of 2
            {
                let rhs_reg = self.expression(expr)?;
                self.insert(Instruction::Slli(
                    rhs_reg.into(),
                    rhs_reg.into(),
                    // 2 0b0010 -> shift by 1 (2 - 1 = 1 0b0001 -> one 1)
                    // 4 0b0100 -> shift by 2 (4 - 1 = 3 0b0011 -> two 1s)
                    // 8 0b1000 -> shift by 3 (8 - 1 = 7 0b0111 -> three 1s)
                    (value - 1).count_ones() as i64,
                ));
                Some(rhs_reg)
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

    /// Helper function which handles parts of infix expressions.
    fn infix_helper(&mut self, lhs: Register, rhs: Register, op: InfixOp, type_: Type) -> Register {
        // creates the two result registers
        // eventually, just one of the two is used
        let dest_regi = self.get_int_reg();
        let dest_regf = self.get_float_reg();

        match (type_, op) {
            (Type::Int(0), InfixOp::Plus) => {
                self.insert(Instruction::Add(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int(0), InfixOp::Minus) => {
                self.insert(Instruction::Sub(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Char(0), InfixOp::Plus) => {
                self.insert(Instruction::Add(dest_regi, lhs.into(), rhs.into()));

                // TODO: is this really required?
                self.use_reg(dest_regi.into(), Size::Byte);
                let mask = self.get_int_reg();
                self.insert(Instruction::Li(mask, 0x7f));
                self.release_reg(dest_regi.into());

                self.insert(Instruction::And(dest_regi, dest_regi, mask));

                dest_regi.into()
            }
            (Type::Char(0), InfixOp::Minus) => {
                self.insert(Instruction::Sub(dest_regi, lhs.into(), rhs.into()));

                // TODO: is this really required?
                self.use_reg(dest_regi.into(), Size::Byte);
                let mask = self.get_int_reg();
                self.insert(Instruction::Li(mask, 0x7f));
                self.release_reg(dest_regi.into());

                self.insert(Instruction::And(dest_regi, dest_regi, mask));

                dest_regi.into()
            }
            (Type::Int(0), InfixOp::Mul) => {
                self.insert(Instruction::Mul(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int(0), InfixOp::Div) => {
                self.insert(Instruction::Div(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int(0), InfixOp::Rem) => {
                self.insert(Instruction::Rem(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int(0), InfixOp::Pow) => self
                .__rush_internal_pow_int(lhs.into(), rhs.into())
                .to_reg(),
            (Type::Int(0), InfixOp::Shl) => {
                self.insert(Instruction::Sll(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int(0), InfixOp::Shr) => {
                self.insert(Instruction::Sra(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int(0) | Type::Bool(0), InfixOp::BitOr | InfixOp::Or) => {
                self.insert(Instruction::Or(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int(0) | Type::Bool(0), InfixOp::BitAnd | InfixOp::And) => {
                self.insert(Instruction::And(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int(0) | Type::Bool(0), InfixOp::BitXor) => {
                self.insert(Instruction::Xor(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (
                // even if not all ops are allowed for char and bool, the analyzer would not accept
                // illegal programs, therefore this is ok.
                Type::Int(0) | Type::Char(0) | Type::Bool(0),
                InfixOp::Eq
                | InfixOp::Neq
                | InfixOp::Lt
                | InfixOp::Lte
                | InfixOp::Gt
                | InfixOp::Gte,
            ) => {
                self.insert(Instruction::SetIntCondition(
                    Condition::from(op),
                    dest_regi,
                    lhs.into(),
                    rhs.into(),
                ));
                dest_regi.to_reg()
            }
            (
                Type::Float(0),
                InfixOp::Eq
                | InfixOp::Neq
                | InfixOp::Lt
                | InfixOp::Lte
                | InfixOp::Gt
                | InfixOp::Gte,
            ) => {
                self.insert(Instruction::SetFloatCondition(
                    Condition::from(op),
                    dest_regi,
                    lhs.into(),
                    rhs.into(),
                ));
                dest_regi.into()
            }
            (Type::Float(0), InfixOp::Plus) => {
                self.insert(Instruction::Fadd(dest_regf, lhs.into(), rhs.into()));
                dest_regf.into()
            }
            (Type::Float(0), InfixOp::Minus) => {
                self.insert(Instruction::Fsub(dest_regf, lhs.into(), rhs.into()));
                dest_regf.into()
            }
            (Type::Float(0), InfixOp::Mul) => {
                self.insert(Instruction::Fmul(dest_regf, lhs.into(), rhs.into()));
                dest_regf.into()
            }
            (Type::Float(0), InfixOp::Div) => {
                self.insert(Instruction::Fdiv(dest_regf, lhs.into(), rhs.into()));
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

        let assignee = self.resolve_name(node.assignee).clone();

        let mut src_ptr = assignee.value.clone();
        let mut assignee_type = assignee.type_;
        let ptr_reg = self.get_int_reg();

        // if the lhs is an indirected pointer, perform required indirections
        if node.assignee_ptr_count > 0 {
            let mut ptr_count = node.assignee_ptr_count;
            // TODO: is this really required?
            self.use_reg(ptr_reg.into(), Size::Dword);

            while ptr_count > 0 {
                self.insert_with_comment(
                    Instruction::Ld(
                        ptr_reg,
                        src_ptr
                            .clone()
                            .expect("the analyzer guarantees valid usage of pointers"),
                    ),
                    "deref".into(),
                );
                src_ptr = Some(Pointer::Register(ptr_reg, 0));
                assignee_type = assignee_type
                    .sub_deref()
                    .expect("the analyzer guarantees valid usage of pointers");
                ptr_count -= 1;
            }
        }

        // holds the value of the rhs (either simple or the result of an operation)
        'outer: {
            let rhs_reg = match node.op {
                AssignOp::Basic => match self.expression(node.expr) {
                    Some(reg) => reg,
                    None => return,
                },
                AssignOp::Pow => {
                    // load value from the lhs
                    let lhs = self
                        // `clone` only clones a [`Rc`]
                        .load_value_from_pointer(
                            src_ptr.clone().expect("the lhs is always a value"),
                            assignee_type,
                            node.assignee,
                        );
                    self.use_reg(lhs, Size::from(assignee_type));

                    // compile the rhs
                    let Some(rhs) = self.expression(node.expr) else { break 'outer };
                    self.use_reg(rhs, Size::from(rhs_type));

                    // call the `pow` corelib function using the `infix_helper`
                    let res = self.infix_helper(lhs, rhs, InfixOp::from(node.op), assignee_type);

                    self.release_reg(lhs);
                    self.release_reg(rhs);
                    res
                }
                _ => {
                    // compile the rhs
                    let Some(rhs) = self.expression(node.expr) else { break 'outer};
                    self.use_reg(rhs, Size::from(rhs_type));

                    // load value from the lhs
                    let lhs = self
                        // `clone` only clones a [`Rc`]
                        .load_value_from_pointer(
                            src_ptr.clone().expect("the lhs is always a value"),
                            assignee_type,
                            node.assignee,
                        );
                    self.use_reg(lhs, Size::from(assignee_type));

                    // perform pre-assign operation using the infix helper
                    let res = self.infix_helper(lhs, rhs, InfixOp::from(node.op), assignee_type);

                    self.release_reg(lhs);
                    self.release_reg(rhs);
                    res
                }
            };

            if let Some(ptr) = src_ptr {
                match rhs_type {
                    Type::Float(0) => self.insert(Instruction::Fsd(rhs_reg.into(), ptr)),
                    Type::Bool(0) | Type::Char(0) => {
                        self.insert(Instruction::Sb(rhs_reg.into(), ptr))
                    }
                    Type::Int(_) | Type::Bool(_) | Type::Char(_) | Type::Float(_) => {
                        self.insert(Instruction::Sd(rhs_reg.into(), ptr))
                    }
                    Type::Unit | Type::Never => {} // ignore these types
                    Type::Unknown => unreachable!("the analyzer would have failed"),
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
                self.insert(Instruction::CastIntToFloat(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Char(0) | Type::Bool(0), Type::Float(0)) => {
                let dest_reg = self.get_float_reg();
                self.insert(Instruction::CastByteToFloat(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Int(0) | Type::Char(0), Type::Bool(0)) => {
                let dest_reg = self.get_int_reg();
                self.insert(Instruction::Snez(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Int(0), Type::Char(0)) => self
                .__rush_internal_cast_int_to_char(lhs_reg.into())
                .to_reg(),
            // float base type casts
            (Type::Float(0), Type::Int(0)) => {
                let dest_reg = self.get_int_reg();
                self.insert(Instruction::CastFloatToInt(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Float(0), Type::Char(0)) => self
                .__rush_internal_cast_float_to_char(lhs_reg.into())
                .to_reg(),
            (Type::Float(0), Type::Bool(0)) => {
                // get a `.rodata` label which holds a float zero to compare to
                let float_zero_label = match self
                    .rodata_section
                    .iter()
                    .find(|o| o.data == DataObjType::Float(0.0))
                {
                    Some(obj) => Rc::clone(&obj.label),
                    None => {
                        // create a float constant with the value 0
                        let label = format!("float_constant_{}", self.rodata_section.len()).into();
                        self.rodata_section.push(DataObj {
                            label: Rc::clone(&label),
                            data: DataObjType::Float(0.0),
                        });
                        label
                    }
                };

                // load value from float constant into a free float register
                let zero_float_reg = self.get_float_reg();
                self.insert(Instruction::Fld(
                    zero_float_reg,
                    Pointer::Label(float_zero_label),
                ));

                // compare the float to `0.0`
                let dest_reg = self.get_int_reg();
                self.insert(Instruction::SetFloatCondition(
                    Condition::Ne,
                    dest_reg,
                    zero_float_reg,
                    lhs_reg.into(),
                ));

                // return the result of the comparison
                dest_reg.to_reg()
            }
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
        let cond_reg = self.expression(node.cond)?;

        // will later hold the result of the branch
        let res_reg = match node.result_type {
            Type::Float(0) => Some(self.get_float_reg().to_reg()),
            Type::Int(0) | Type::Bool(0) | Type::Char(0) => Some(self.get_int_reg().to_reg()),
            _ => None, // other types require no register
        };

        let merge_block = self.gen_label("merge");
        let else_block_label = self.gen_label("else");

        // if the condition evaluated to `false`, jump to the `else` or `merge` block
        self.insert(Instruction::BrCond(
            Condition::Eq,
            cond_reg.into(),
            IntRegister::Zero,
            if node.else_block.is_some() {
                Rc::clone(&else_block_label)
            } else {
                Rc::clone(&merge_block)
            },
        ));

        let then_reg = self.block(node.then_block);

        // if the `then` block returns a register other than res, move the block register into res
        match (res_reg, then_reg) {
            (Some(Register::Int(res)), Some(Register::Int(then_reg))) => {
                self.insert(Instruction::Mv(res, then_reg));
            }
            (Some(Register::Float(res)), Some(Register::Float(then_reg))) => {
                self.insert(Instruction::Fmv(res, then_reg));
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
                    self.insert(Instruction::Mv(res, else_reg));
                }
                (Some(Register::Float(res)), Some(Register::Float(else_reg))) => {
                    self.insert(Instruction::Fmv(res, else_reg));
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

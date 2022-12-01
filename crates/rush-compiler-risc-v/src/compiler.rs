use std::collections::HashMap;

use rush_analyzer::{ast::*, AssignOp, InfixOp, PrefixOp, Type};

use crate::{
    instruction::{Block, Condition, Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register},
    utils::{DataObj, DataObjType, Function, Loop, Size, Variable, VariableValue},
};

pub struct Compiler<'tree> {
    /// Specifies all exported labels of the program.
    pub(crate) exports: Vec<String>,

    /// Labels and their basic blocks which contain instructions.
    pub(crate) blocks: Vec<Block>,
    /// Points to the current section which is inserted to.
    pub(crate) curr_block: usize,

    /// Data section for storing global variables.
    pub(crate) data_section: Vec<DataObj>,
    /// Read-only data section for storing constant values (like floats).
    pub(crate) rodata_section: Vec<DataObj>,

    /// Holds metadata about the current function
    pub(crate) curr_fn: Option<Function>,
    /// Holds metadata about the current loop
    pub(crate) curr_loop: Option<Loop>,

    /// The first element is the root scope, the last element is the current scope.
    pub(crate) scopes: Vec<HashMap<&'tree str, Variable>>,
    /// Holds the global variables of the program.
    pub(crate) globals: HashMap<&'tree str, Variable>,

    /// Specifies all registers which are currently in use and may not be overwritten.
    pub(crate) used_registers: Vec<(Register, Size)>,
}

impl<'tree> Compiler<'tree> {
    /// Creates and returns a new [`Compiler`].
    pub fn new() -> Self {
        Self {
            blocks: vec![],
            exports: vec![],
            curr_block: 0,
            data_section: vec![],
            rodata_section: vec![],
            scopes: vec![],
            globals: HashMap::new(),
            curr_fn: None,
            curr_loop: None,
            used_registers: vec![],
        }
    }

    /// Compiles the source AST into a RISC-V targeted Assembly program.
    pub fn compile(&mut self, ast: &'tree AnalyzedProgram) -> String {
        self.declare_main_fn(&ast.main_fn, &ast.globals);

        for func in ast.functions.iter().filter(|f| f.used) {
            self.function_declaration(func)
        }

        self.codegen()
    }

    /// Generates the Assembly representation from the compiled program.
    /// This function is invoked in the last step of compilation and only generates the output.
    fn codegen(&self) -> String {
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
            .map(|d| d.to_string())
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

        // declares constants (like floats) under the `.rodata` section
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

    /// Compiles the `main` function and the global variables of the program.
    fn declare_main_fn(
        &mut self,
        node: &'tree AnalyzedBlock,
        globals: &'tree Vec<AnalyzedLetStmt>,
    ) {
        // create `_start` label
        let start_label = self.append_block("_start");
        let fn_block = self.append_block("main..main");
        self.exports.push(start_label);

        // call the main function
        self.insert(Instruction::Call(fn_block.clone()));

        // default: exit with code 0
        self.insert(Instruction::Li(IntRegister::A0, 0));
        self.insert(Instruction::Call("exit".into()));

        // declare global variables
        // can be declared before the prologue: exprs are all constant (require no stack)
        for var in globals {
            self.declare_global(var.name, var.mutable, &var.expr)
        }

        // add the epilogue label
        let epilogue_label = self.gen_label("epilogue");
        self.curr_fn = Some(Function::new(epilogue_label.clone()));

        // compile the function body
        self.push_scope();
        self.insert_at(&fn_block);
        self.function_body(node, epilogue_label.clone());
        self.pop_scope();

        // prologue is inserted after the body (because it now knows about the frame size)
        let mut prologue = self.prologue();
        self.insert_at(&fn_block); // resets the current block back to the fn block
        prologue.append(&mut self.blocks[self.curr_block].instructions);
        self.blocks[self.curr_block].instructions = prologue;

        self.blocks.push(Block::new(epilogue_label.clone()));
        self.insert_at(&epilogue_label);
        self.epilogue()
    }

    /// Declares a new global variable.
    /// The initializer is put inside the current basic block.
    /// If the variable is non-mutable, it is put under the `.rodata` section
    fn declare_global(
        &mut self,
        label: &'tree str,
        mutable: bool,
        value: &'tree AnalyzedExpression,
    ) {
        let type_ = value.result_type();
        let data = match (type_, value) {
            (Type::Int, AnalyzedExpression::Int(val)) => DataObjType::Dword(*val),
            (Type::Bool, AnalyzedExpression::Bool(val)) => DataObjType::Byte(*val as i64),
            (Type::Char, AnalyzedExpression::Char(val)) => DataObjType::Byte(*val as i64),
            (Type::Float, AnalyzedExpression::Float(val)) => DataObjType::Float(*val),
            _ => unreachable!("other types cannot be used as globals"),
        };

        let value = match mutable {
            true => {
                self.data_section.push(DataObj {
                    label: label.to_string(),
                    data,
                });
                VariableValue::Pointer(Pointer::Label(label.to_string()))
            }
            false => {
                let value = VariableValue::Pointer(
                    match self.rodata_section.iter().find(|d| d.data == data) {
                        Some(DataObj { label, .. }) => Pointer::Label(label.to_string()),
                        None => {
                            self.rodata_section.push(DataObj {
                                label: label.to_string(),
                                data,
                            });
                            Pointer::Label(label.to_string())
                        }
                    },
                );
                value
            }
        };

        self.globals.insert(label, Variable { type_, value });
    }

    /// Compiles an [`AnalyzedFunctionDefinition`] declaration.
    fn function_declaration(&mut self, node: &'tree AnalyzedFunctionDefinition) {
        // append block for the function
        let fn_block = format!("main..{}", node.name);
        self.append_block(&fn_block);
        self.insert_at(&fn_block);

        // add the epilogue label
        let epilogue_label = self.gen_label("epilogue");
        self.curr_fn = Some(Function::new(epilogue_label.clone()));

        // push a new scope for the function
        self.push_scope();

        // specifies which param is the current one ( 0 = a0 / fa0)
        let mut int_cnt = 0;
        let mut float_cnt = 0;

        // specifies the memory offset to use when params are spilled
        // is incremented in steps of 8
        let mut mem_offset = 0;

        let mut param_store_instructions = vec![(
            Instruction::Comment("save params on stack".to_string()),
            None,
        )];

        // save all param values in the current scope / on the stack
        for param in &node.params {
            match param.type_ {
                Type::Int | Type::Char | Type::Bool => {
                    match IntRegister::nth_param(int_cnt) {
                        Some(reg) => {
                            let size = Size::from(param.type_);
                            Self::align(&mut self.curr_fn_mut().stack_allocs, size.byte_count());
                            self.curr_fn_mut().stack_allocs += size.byte_count();
                            let offset = -self.curr_fn().stack_allocs as i64 - 16;

                            // use `sb` or `sd` depending on the size
                            match size {
                                Size::Byte => param_store_instructions.push((
                                    Instruction::Sb(reg, Pointer::Stack(IntRegister::Fp, offset)),
                                    Some(format!("param {} = {reg}", param.name)),
                                )),
                                Size::Dword => param_store_instructions.push((
                                    Instruction::Sd(reg, Pointer::Stack(IntRegister::Fp, offset)),
                                    Some(format!("param {} = {reg}", param.name)),
                                )),
                            }

                            self.scope_mut().insert(
                                param.name,
                                Variable {
                                    type_: param.type_,
                                    value: VariableValue::Pointer(Pointer::Stack(
                                        IntRegister::Fp,
                                        offset,
                                    )),
                                },
                            );
                        }
                        None => {
                            self.scope_mut().insert(
                                param.name,
                                Variable {
                                    type_: param.type_,
                                    value: VariableValue::Pointer(Pointer::Stack(
                                        IntRegister::Fp,
                                        mem_offset,
                                    )),
                                },
                            );
                            mem_offset += 8;
                        }
                    }
                    int_cnt += 1;
                }
                Type::Float => {
                    match FloatRegister::nth_param(float_cnt) {
                        Some(reg) => {
                            let size = Size::from(param.type_).byte_count();
                            Self::align(&mut self.curr_fn_mut().stack_allocs, size);
                            self.curr_fn_mut().stack_allocs += size;
                            let offset = -self.curr_fn().stack_allocs as i64 - 16;

                            param_store_instructions.push((
                                Instruction::Fsd(reg, Pointer::Stack(IntRegister::Fp, offset)),
                                Some(format!("param {} = {reg}", param.name)),
                            ));

                            self.scope_mut().insert(
                                param.name,
                                Variable {
                                    type_: param.type_,
                                    value: VariableValue::Pointer(Pointer::Stack(
                                        IntRegister::Fp,
                                        offset,
                                    )),
                                },
                            );
                        }
                        None => {
                            self.scope_mut().insert(
                                param.name,
                                Variable {
                                    type_: param.type_,
                                    value: VariableValue::Pointer(Pointer::Stack(
                                        IntRegister::Fp,
                                        mem_offset,
                                    )),
                                },
                            );
                            mem_offset += 8;
                        }
                    }
                    float_cnt += 1;
                }
                Type::Unit | Type::Never => {
                    // add dummy values for these types
                    self.scope_mut().insert(
                        param.name,
                        Variable {
                            type_: Type::Unit,
                            value: VariableValue::Unit,
                        },
                    );
                }
                Type::Unknown => unreachable!("analyzer would have failed"),
            }
        }

        // compile the function body
        self.function_body(&node.block, epilogue_label.clone());
        self.pop_scope();

        // compile prologue
        let mut prologue = self.prologue();
        self.insert_at(&fn_block); // resets the current block back to the fn block
        prologue.append(&mut param_store_instructions);
        prologue.append(&mut self.blocks[self.curr_block].instructions);
        self.blocks[self.curr_block].instructions = prologue;

        // compile epilogue
        self.blocks.push(Block::new(epilogue_label));

        // generate epilogue
        self.epilogue()
    }

    /// Compiles the body of a function.
    fn function_body(&mut self, node: &'tree AnalyzedBlock, epilogue_label: String) {
        // add debugging comment
        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment("begin body".to_string()));

        // compile each statement
        for stmt in &node.stmts {
            self.statement(stmt);
        }

        // place the result of the optional expression in a return value register(s)
        // for `int`, `bool`, and `char`: `a0`
        // for `float`: `fa0`
        if let Some(expr) = &node.expr {
            let res_reg = self.expression(expr);
            match res_reg {
                Some(Register::Int(reg)) => {
                    self.insert(Instruction::Mov(IntRegister::A0, reg));
                }
                Some(Register::Float(reg)) => {
                    self.insert(Instruction::Fmv(FloatRegister::Fa0, reg));
                }
                None => {} // do nothing with unit values
            }
        }

        // add debugging comment
        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment("end body".to_string()));

        // jump to the epilogue
        self.insert_jmp(epilogue_label);
    }

    /// Compiles an [`AnalyzedBlock`].
    /// Automatically pushes a new scope for the block.
    fn block(&mut self, node: &'tree AnalyzedBlock) -> Option<Register> {
        // push a new scope
        self.push_scope();

        for stmt in &node.stmts {
            self.statement(stmt)
        }

        // return expression register if there is an expr
        let res = node.expr.as_ref().and_then(|e| self.expression(e));

        // pop the scope again
        self.pop_scope();

        res
    }

    /// Copiles an [`AnalyzedStatement`].
    /// Invokes the corresponding function for most of the statement options.
    fn statement(&mut self, node: &'tree AnalyzedStatement) {
        match node {
            AnalyzedStatement::Let(node) => self.let_statement(node),
            AnalyzedStatement::Return(node) => self.return_stmt(node),
            AnalyzedStatement::Loop(node) => self.loop_stmt(node),
            AnalyzedStatement::While(node) => self.while_stmt(node),
            AnalyzedStatement::For(node) => self.for_stmt(node),
            AnalyzedStatement::Break => {
                #[cfg(debug_assertions)]
                self.insert(Instruction::Comment("break".to_string()));
                self.insert_jmp(self.curr_loop().after_loop.clone())
            }
            AnalyzedStatement::Continue => {
                #[cfg(debug_assertions)]
                self.insert(Instruction::Comment("continue".to_string()));
                self.insert_jmp(self.curr_loop().loop_head.clone())
            }
            AnalyzedStatement::Expr(node) => {
                self.expression(node);
            }
        }
    }

    /// Compiles an [`AnalyzedReturnStmt`].
    /// If the node contains an optional expr, it is compiled and its result is moved into the
    /// correct return-value register (corresponding to the result type of the expr).
    fn return_stmt(&mut self, node: &'tree AnalyzedReturnStmt) {
        // if there is an expression, compile it
        if let Some(expr) = node {
            match self.expression(expr) {
                Some(Register::Int(reg)) => self.insert(Instruction::Mov(IntRegister::A0, reg)),
                Some(Register::Float(reg)) => {
                    self.insert(Instruction::Fmv(FloatRegister::Fa0, reg))
                }
                None => {} // returns unit, do nothing
            }
        }
        // jump to the epilogue label (do actual return)
        self.insert_jmp(self.curr_fn().epilogue_label.clone());
    }

    /// Compiles an [`AnalyzedLoopStmt`].
    /// After each iteration, there is an unconditional jump back to the loop head (i.e. `continue`).
    /// In this looping construct, manual control flow like `break` is mandatory.
    fn loop_stmt(&mut self, node: &'tree AnalyzedLoopStmt) {
        let loop_head = self.append_block("loop_head");
        let after_loop = self.gen_label("after_loop");
        self.curr_loop = Some(Loop {
            loop_head: loop_head.clone(),
            after_loop: after_loop.clone(),
        });

        self.insert_at(&loop_head);
        self.block(&node.block);
        self.insert_jmp(loop_head);

        self.blocks.push(Block::new(after_loop.clone()));
        self.insert_at(&after_loop);
    }

    /// Compiles an [`AnalyzedWhileStmt`].
    /// Before each iteration, the loop condition is evaluated and compared against `false`.
    /// If the result is `false`, there is a jump to the basic block after the loop (i.e. `break`).
    fn while_stmt(&mut self, node: &'tree AnalyzedWhileStmt) {
        let while_head = self.append_block("while_head");
        let after_loop = self.gen_label("after_while");
        self.insert_at(&while_head);

        // compile the condition
        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment("while condition".to_string()));

        let expr_res = self.expression(&node.cond).expect("cond is not unit");
        // if the condition evaluates to `false`, break out of the loop
        self.insert(Instruction::BrCond(
            Condition::Eq,
            IntRegister::Zero,
            expr_res.into(),
            after_loop.clone(),
        ));

        // compile the body
        self.curr_loop = Some(Loop {
            loop_head: while_head.clone(),
            after_loop: after_loop.clone(),
        });

        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment("while body".to_string()));

        self.block(&node.block);
        // jump back to the loop head
        self.insert_jmp(while_head);

        self.blocks.push(Block::new(after_loop.clone()));

        // place the cursor after the loop body
        self.insert_at(&after_loop);
    }

    /// Compiles an [`AnalyzedForStmt`].
    /// Before the loop starts, an induction variable is initialized to a value.
    /// Before each iteration, the loop condition is verified to be `true`.
    /// Otherwise, there will be a `break` out of the looping construct.
    /// At the end of each iteration, the update expression is invoked, its value is omitted.
    fn for_stmt(&mut self, node: &'tree AnalyzedForStmt) {
        let for_head = self.append_block("for_head");
        let after_loop = self.append_block("after_for");

        //// INIT ////
        self.insert(Instruction::Comment(format!("loop init: {}", node.ident)));

        let type_ = node.initializer.result_type();
        let res = self.expression(&node.initializer).map(|reg| {
            self.use_reg(reg, Size::from(type_));
            (
                Variable {
                    type_,
                    value: VariableValue::Register(reg),
                },
                reg,
            )
        });

        // add a new scope and insert the induction variable into it
        self.push_scope();

        self.scope_mut().insert(
            node.ident,
            match res {
                Some((ref var, _)) => var.clone(),
                None => Variable::unit(),
            },
        );

        //// CONDITION ////
        self.insert_at(&for_head);

        self.insert(Instruction::Comment("loop condition".to_string()));

        let expr_res = self.expression(&node.cond).expect("cond is non-unit");
        self.insert(Instruction::BrCond(
            Condition::Eq,
            IntRegister::Zero,
            expr_res.into(),
            after_loop.clone(),
        ));

        //// BODY ////
        self.insert(Instruction::Comment("loop body".to_string()));

        self.curr_loop = Some(Loop {
            loop_head: for_head.clone(),
            after_loop: after_loop.clone(),
        });

        // compile the loop block: cannot use `self.block` due to scoping
        for stmt in &node.block.stmts {
            self.statement(stmt)
        }
        // compile optional expr
        //node.block.expr.map(|e| self.expression(&e));

        if let Some(expr) = &node.block.expr {
            self.expression(expr);
        };

        //// UPDATE EXPR ////
        self.insert(Instruction::Comment("loop update".to_string()));

        self.expression(&node.update);
        self.pop_scope();

        // jump back to the loop start
        self.insert_jmp(for_head);

        //// TAIL ////
        self.insert_at(&after_loop);

        // release induction register at the end
        if let Some(reg) = res.map(|r| r.1) {
            self.release_reg(reg)
        }
    }

    /// Compiles an [`AnalyzedLetStmt`]
    /// Allocates a new variable on the stack.
    /// Also increments the `stack_allocs` value of the current function
    fn let_statement(&mut self, node: &'tree AnalyzedLetStmt) {
        let type_ = node.expr.result_type();

        // filter out any unit / never types
        let rhs_reg = match self.expression(&node.expr) {
            Some(reg) => reg,
            None => {
                // unit / never type: insert a dummy variable into the HashMap
                self.scope_mut().insert(node.name, Variable::unit());
                return;
            }
        };

        // alignment and offset calculation
        let size = Size::from(type_).byte_count();
        Self::align(&mut self.curr_fn_mut().stack_allocs, size);
        self.curr_fn_mut().stack_allocs += size as i64;
        let offset = -self.curr_fn().stack_allocs as i64 - 16;

        let comment = format!("let {} = {rhs_reg}", node.name,);
        match rhs_reg {
            Register::Int(reg) => match type_ {
                Type::Bool | Type::Char => self.insert_w_comment(
                    Instruction::Sb(reg, Pointer::Stack(IntRegister::Fp, offset)),
                    comment,
                ),
                Type::Int => self.insert_w_comment(
                    Instruction::Sd(reg, Pointer::Stack(IntRegister::Fp, offset)),
                    comment,
                ),
                _ => unreachable!("only the types above use int registers"),
            },
            Register::Float(reg) => self.insert_w_comment(
                Instruction::Fsd(reg, Pointer::Stack(IntRegister::Fp, offset)),
                comment,
            ),
        }

        // insert variable into the current scope
        let var = Variable {
            type_,
            value: VariableValue::Pointer(Pointer::Stack(IntRegister::Fp, offset)),
        };
        self.scope_mut().insert(node.name, var);
    }

    /// Compiles an [`AnalyzedExpression`].
    pub(crate) fn expression(&mut self, node: &'tree AnalyzedExpression) -> Option<Register> {
        match node {
            AnalyzedExpression::Int(value) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Li(dest_reg, *value));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Bool(value) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Li(dest_reg, *value as i64));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Char(value) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Li(dest_reg, *value as i64));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Float(value) => {
                let dest_reg = self.alloc_freg();

                // if there is already a float constant with this value, use its label
                // otherwise, create a new float constant and use its label
                let float_value_label = match self
                    .rodata_section
                    .iter()
                    .find(|o| o.data == DataObjType::Float(*value))
                {
                    Some(obj) => obj.label.clone(),
                    None => {
                        let label = format!("float_constant_{}", self.rodata_section.len());
                        self.rodata_section.push(DataObj {
                            label: label.clone(),
                            data: DataObjType::Float(*value),
                        });
                        label
                    }
                };

                // load the value from float constant into `dest_reg`
                self.insert(Instruction::Fld(
                    dest_reg,
                    Pointer::Label(float_value_label),
                ));

                Some(Register::Float(dest_reg))
            }
            AnalyzedExpression::Ident(ident) => {
                // if this is a placeholder or dummy variable, ignore it
                let var = match self.resolve_name(ident.ident).clone() {
                    Variable { value, .. } if value == VariableValue::Unit => return None,
                    var => var,
                };
                self.load_value_from_variable(var, ident.ident.to_string())
            }
            AnalyzedExpression::Prefix(node) => self.prefix_expr(node),
            AnalyzedExpression::Infix(node) => self.infix_expr(node),
            AnalyzedExpression::Assign(node) => {
                self.assign_expr(node);
                None
            }
            AnalyzedExpression::Call(node) => self.call_expr(node),
            AnalyzedExpression::Cast(node) => self.cast_expr(node),
            AnalyzedExpression::Grouped(node) => self.expression(node),
            AnalyzedExpression::Block(node) => self.block(node),
            AnalyzedExpression::If(node) => self.if_expr(node),
        }
    }

    /// Loads the specified variable into a register.
    /// Decides which load operation is to be used as it depends on the data size.
    fn load_value_from_variable(&mut self, var: Variable, ident: String) -> Option<Register> {
        match var.value {
            VariableValue::Pointer(ptr) => match var.type_ {
                Type::Bool | Type::Char => {
                    let dest_reg = self.alloc_ireg();
                    self.insert_w_comment(Instruction::Lb(dest_reg, ptr), ident);
                    Some(Register::Int(dest_reg))
                }
                Type::Int => {
                    let dest_reg = self.alloc_ireg();
                    self.insert_w_comment(Instruction::Ld(dest_reg, ptr), ident);
                    Some(Register::Int(dest_reg))
                }
                Type::Float => {
                    let dest_reg = self.alloc_freg();
                    self.insert_w_comment(Instruction::Fld(dest_reg, ptr), ident);
                    Some(Register::Float(dest_reg))
                }
                _ => unreachable!("either filtered or impossible"),
            },
            VariableValue::Register(reg) => Some(reg),
            VariableValue::Unit => None,
        }
    }

    /// Compiles an [`AnalyzedAssignExpr`].
    /// Performs simple assignments and complex operator-backed assignments.
    /// For the latter, the assignee's current value is loaded into a temporary register.
    /// Following that, the operation is performed by `self.infix_helper`.
    /// Lastly, a correct store instruction is used to assign the resulting value to the assignee.
    fn assign_expr(&mut self, node: &'tree AnalyzedAssignExpr) {
        let rhs_type = node.expr.result_type();

        let assignee = self.resolve_name(node.assignee).clone();

        // holds the value of the rhs (either simple or the result of an operation)
        let rhs_reg = match node.op {
            AssignOp::Basic => match self.expression(&node.expr) {
                Some(reg) => reg,
                None => return,
            },
            AssignOp::Pow => {
                // load value from the lhs
                let lhs = self
                    .load_value_from_variable(assignee.clone(), node.assignee.to_string())
                    .expect("filtered above");
                self.use_reg(lhs, Size::from(assignee.type_));

                // compile the rhs
                let Some(rhs) = self.expression(&node.expr) else {return };
                self.use_reg(rhs, Size::from(rhs_type));

                // call the `pow` corelib function using the `infix_helper`
                let res = self.infix_helper(lhs, rhs, InfixOp::from(node.op), assignee.type_);

                self.release_reg(lhs);
                self.release_reg(rhs);
                res
            }
            _ => {
                // compile the rhs
                let Some(rhs) = self.expression(&node.expr) else {return };
                self.use_reg(rhs, Size::from(rhs_type));

                // load value from the lhs
                let lhs = self
                    .load_value_from_variable(assignee.clone(), node.assignee.to_string())
                    .expect("filtered above");
                self.use_reg(lhs, Size::from(assignee.type_));

                // perform pre-assign operation using the infix helper
                let res = self.infix_helper(lhs, rhs, InfixOp::from(node.op), assignee.type_);

                self.release_reg(lhs);
                self.release_reg(rhs);
                res
            }
        };

        match assignee.value {
            VariableValue::Pointer(ptr) => match rhs_type {
                Type::Int => self.insert(Instruction::Sd(rhs_reg.into(), ptr)),
                Type::Float => self.insert(Instruction::Fsd(rhs_reg.into(), ptr)),
                Type::Bool | Type::Char => self.insert(Instruction::Sb(rhs_reg.into(), ptr)),
                Type::Unit | Type::Never => {} // ignore unit types
                _ => unreachable!("the analyzer would have failed"),
            },
            VariableValue::Register(dest) => match rhs_type {
                Type::Int | Type::Bool | Type::Char => {
                    self.insert(Instruction::Mov((dest).into(), rhs_reg.into()))
                }
                Type::Float => self.insert(Instruction::Fmv(dest.into(), rhs_reg.into())),
                _ => unreachable!("other types cannot exist in an assignment"),
            },
            VariableValue::Unit => {} // do nothing for unit types
        }
    }

    /// Compiles an [`AnalyzedIfExpr`].
    /// The result of the expression is saved in a single register (reflecting the result type).
    /// Control flow is accomplished through the use of branches.
    /// The condition is verified using a normal conditional branch, comparing it to `true`.
    fn if_expr(&mut self, node: &'tree AnalyzedIfExpr) -> Option<Register> {
        // (bool) result of the condition
        let cond_reg = self
            .expression(&node.cond)
            .expect("cond is not unit / never");

        // will later hold the result of this expr
        let res_reg = match node.result_type {
            Type::Float => Some(self.alloc_freg().to_reg()),
            Type::Int | Type::Bool | Type::Char => Some(self.alloc_ireg().to_reg()),
            _ => None, // other types require no register
        };

        let then_block = self.append_block("then");
        let merge_block = self.append_block("merge");

        // if the condition evaluated to `1` / `true`, the `then` block is entered
        self.insert(Instruction::BrCond(
            Condition::Ne,
            cond_reg.into(),
            IntRegister::Zero,
            then_block.clone(),
        ));

        // if there is an `else` block, compile it
        if let Some(else_block) = &node.else_block {
            let else_block_label = self.append_block("else");
            // stands directly below the conditional branch
            self.insert_jmp(else_block_label.clone());
            self.insert_at(&else_block_label);
            let else_reg = self.block(else_block);

            // if the block returns a register other than res, move the block register into res
            match (res_reg, else_reg) {
                (Some(Register::Int(res)), Some(Register::Int(else_reg))) => {
                    self.insert(Instruction::Mov(res, else_reg));
                }
                (Some(Register::Float(res)), Some(Register::Float(else_reg))) => {
                    self.insert(Instruction::Fmv(res, else_reg));
                }
                _ => {}
            }
        }

        // regardless of the previous block, insert a jump to the `merge` block
        self.insert_jmp(merge_block.clone());

        self.insert_at(&then_block);
        let then_reg = self.block(&node.then_block);

        // if the block returns a register other than res, move the block register into res
        match (res_reg, then_reg) {
            (Some(Register::Int(res)), Some(Register::Int(then_reg))) => {
                self.insert(Instruction::Mov(res, then_reg));
            }
            (Some(Register::Float(res)), Some(Register::Float(then_reg))) => {
                self.insert(Instruction::Fmv(res, then_reg));
            }
            _ => {}
        }

        // jump to the `merge` block after this branch
        self.insert_jmp(merge_block.clone());

        // set the cursor position to the end of the `merge` block
        self.insert_at(&merge_block);

        res_reg
    }

    /// Compiles an [`AnalyzedInfixExpr`].
    /// After compiling the lhs and rhs, the `infix_helper` is invoked.
    fn infix_expr(&mut self, node: &'tree AnalyzedInfixExpr) -> Option<Register> {
        match (node.lhs.result_type(), node.op) {
            (Type::Bool, InfixOp::Or) => {
                // compile the lhs (initial expression)
                let lhs_cond = self.expression(&node.lhs)?;

                let merge_block = self.append_block("merge");

                // jump to the merge block if the lhs is true
                self.insert_w_comment(
                    Instruction::BrCond(
                        Condition::Ne,
                        lhs_cond.into(),
                        IntRegister::Zero,
                        merge_block.clone(),
                    ),
                    "||".to_string(),
                );

                // compile the rhs
                let rhs = self.expression(&node.rhs);

                #[cfg(debug_assertions)]
                if let Some(rhs) = rhs {
                    assert_eq!(lhs_cond, rhs);
                }

                self.insert(Instruction::Jmp(merge_block.clone()));

                self.insert_at(&merge_block);

                Some(lhs_cond)
            }
            (Type::Bool, InfixOp::And) => {
                // compile the lhs (initial expression)
                let lhs_cond = self.expression(&node.lhs)?;

                let merge_block = self.append_block("merge");

                // jump to the merge block directly
                self.insert_w_comment(
                    Instruction::BrCond(
                        Condition::Eq,
                        lhs_cond.into(),
                        IntRegister::Zero,
                        merge_block.clone(),
                    ),
                    "&&".to_string(),
                );

                // compile the rhs
                let rhs = self.expression(&node.rhs);

                #[cfg(debug_assertions)]
                if let Some(rhs) = rhs {
                    assert_eq!(lhs_cond, rhs);
                }

                // TODO: if this is broken, uncomment this code
                // if the rhs does not match the lhs, move the rhs into the lhs
                /* match (lhs_cond, rhs) {
                    (Register::Int(lhs), Some(Register::Int(rhs))) if lhs == rhs => {}
                    (Register::Int(lhs), Some(Register::Int(rhs))) => {
                        self.insert(Instruction::Mov(lhs, rhs))
                    }
                    (Register::Float(lhs), Some(Register::Float(rhs))) if lhs == rhs => {}
                    (Register::Float(lhs), Some(Register::Float(rhs))) => {
                        self.insert(Instruction::Fmv(lhs, rhs))
                    }
                    _ => unreachable!("lhs and rhs are always the same type"),
                } */

                self.insert_at(&merge_block);

                Some(lhs_cond)
            }
            _ => {
                let lhs_type = node.lhs.result_type();

                let lhs_reg = self.expression(&node.lhs)?;
                // mark the lhs register as used
                self.use_reg(lhs_reg, Size::from(lhs_type));

                let rhs_type = node.rhs.result_type();
                let rhs_reg = self.expression(&node.rhs)?;
                // mark the rhs register as used
                self.use_reg(rhs_reg, Size::from(rhs_type));

                // release the usage block of the operands
                self.release_reg(lhs_reg);
                self.release_reg(rhs_reg);

                let res = self.infix_helper(lhs_reg, rhs_reg, node.op, lhs_type);

                // TODO: if the above is broken, release the operands here
                // self.release_reg(lhs_reg);
                // self.release_reg(rhs_reg);

                Some(res)
            }
        }
    }

    /// Helper function which handles infix expressions.
    fn infix_helper(&mut self, lhs: Register, rhs: Register, op: InfixOp, type_: Type) -> Register {
        // creates the two result registers
        // eventually, just one of the two is used
        let dest_regi = self.alloc_ireg();
        let dest_regf = self.alloc_freg();

        match (type_, op) {
            (Type::Int, InfixOp::Plus) => {
                self.insert(Instruction::Add(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int, InfixOp::Minus) => {
                self.insert(Instruction::Sub(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Char, InfixOp::Plus) => {
                self.insert(Instruction::Add(dest_regi, lhs.into(), rhs.into()));

                self.use_reg(dest_regi.into(), Size::Byte);
                let mask = self.alloc_ireg();
                self.insert(Instruction::Li(mask, 0x7f));
                self.release_reg(dest_regi.into());

                self.insert(Instruction::And(dest_regi, dest_regi, mask));

                dest_regi.into()
            }
            (Type::Char, InfixOp::Minus) => {
                self.insert(Instruction::Sub(dest_regi, lhs.into(), rhs.into()));

                self.use_reg(dest_regi.into(), Size::Byte);
                let mask = self.alloc_ireg();
                self.insert(Instruction::Li(mask, 0x7f));
                self.release_reg(dest_regi.into());

                self.insert(Instruction::And(dest_regi, dest_regi, mask));

                dest_regi.into()
            }
            (Type::Int, InfixOp::Mul) => {
                self.insert(Instruction::Mul(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int, InfixOp::Div) => {
                self.insert(Instruction::Div(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int, InfixOp::Rem) => {
                self.insert(Instruction::Rem(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int, InfixOp::Pow) => self
                .__rush_internal_pow_int(lhs.into(), rhs.into())
                .to_reg(),
            (Type::Int, InfixOp::Shl) => {
                self.insert(Instruction::Sl(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int, InfixOp::Shr) => {
                self.insert(Instruction::Sr(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int | Type::Bool, InfixOp::BitOr | InfixOp::Or) => {
                self.insert(Instruction::Or(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int | Type::Bool, InfixOp::BitAnd | InfixOp::And) => {
                self.insert(Instruction::And(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (Type::Int | Type::Bool, InfixOp::BitXor) => {
                self.insert(Instruction::Xor(dest_regi, lhs.into(), rhs.into()));
                dest_regi.into()
            }
            (
                // even if not all ops are allowed for char and bool, the analyzer would not accept
                // illegal programs, therefore this is ok.
                Type::Int | Type::Char | Type::Bool,
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
                Type::Float,
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
            (Type::Float, InfixOp::Plus) => {
                self.insert(Instruction::Fadd(dest_regf, lhs.into(), rhs.into()));
                dest_regf.into()
            }
            (Type::Float, InfixOp::Minus) => {
                self.insert(Instruction::Fsub(dest_regf, lhs.into(), rhs.into()));
                dest_regf.into()
            }
            (Type::Float, InfixOp::Mul) => {
                self.insert(Instruction::Fmul(dest_regf, lhs.into(), rhs.into()));
                dest_regf.into()
            }
            (Type::Float, InfixOp::Div) => {
                self.insert(Instruction::Fdiv(dest_regf, lhs.into(), rhs.into()));
                dest_regf.into()
            }
            _ => unreachable!("the analyzer does not allow other combinations"),
        }
    }

    /// Compiles an [`AnalyzedPrefixExpr`].
    fn prefix_expr(&mut self, node: &'tree AnalyzedPrefixExpr) -> Option<Register> {
        let lhs_type = node.expr.result_type();
        let lhs_reg = self.expression(&node.expr)?;

        match (lhs_type, node.op) {
            (Type::Int, PrefixOp::Neg) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Neg(dest_reg, lhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Float, PrefixOp::Neg) => {
                let dest_reg = self.alloc_freg();
                self.insert(Instruction::FNeg(dest_reg, lhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Int, PrefixOp::Not) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Not(dest_reg, lhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Bool, PrefixOp::Not) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Seqz(dest_reg, lhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            _ => unreachable!("other combinations cannot occur in prefix expressions"),
        }
    }

    /// Compiles an [`AnalyzedCastExpr`].
    /// When casting to `char` values, cast functions from the `corelib` are invoked.
    fn cast_expr(&mut self, node: &'tree AnalyzedCastExpr) -> Option<Register> {
        let lhs_type = node.expr.result_type();
        let lhs_reg = self.expression(&node.expr)?;

        // block the use of the lhs temporarily
        self.use_reg(lhs_reg, Size::from(lhs_type));

        let res = match (lhs_type, node.type_) {
            // nop: just return the lhs
            (lhs, rhs) if lhs == rhs => lhs_reg,
            (Type::Bool, Type::Int) | (Type::Bool, Type::Char) | (Type::Char, Type::Int) => lhs_reg,
            // integer base type casts
            (Type::Int, Type::Float) => {
                let dest_reg = self.alloc_freg();
                self.insert(Instruction::CastIntToFloat(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Char | Type::Bool, Type::Float) => {
                let dest_reg = self.alloc_freg();
                self.insert(Instruction::CastByteToFloat(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Int | Type::Char, Type::Bool) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Snez(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Int, Type::Char) => self
                .__rush_internal_cast_int_to_char(lhs_reg.into())
                .to_reg(),
            // float base type casts
            (Type::Float, Type::Int) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::CastFloatToInt(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Float, Type::Char) => self
                .__rush_internal_cast_float_to_char(lhs_reg.into())
                .to_reg(),
            (Type::Float, Type::Bool) => {
                // get a `.rodata` label which holds a float zero to compare to
                let float_zero_label = match self
                    .rodata_section
                    .iter()
                    .find(|o| o.data == DataObjType::Float(0.0))
                {
                    Some(obj) => obj.label.clone(),
                    None => {
                        // create a float constant with the value 0
                        let label = format!("float_constant_{}", self.rodata_section.len());
                        self.rodata_section.push(DataObj {
                            label: label.clone(),
                            data: DataObjType::Float(0.0),
                        });
                        label
                    }
                };

                // load value from float constant into a free float register
                let zero_float_reg = self.alloc_freg();
                self.insert(Instruction::Fld(
                    zero_float_reg,
                    Pointer::Label(float_zero_label),
                ));

                // compare the float to `0.0`
                let dest_reg = self.alloc_ireg();
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
}

impl<'tree> Default for Compiler<'tree> {
    fn default() -> Self {
        Self::new()
    }
}

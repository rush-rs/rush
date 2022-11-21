use std::collections::HashMap;

use rush_analyzer::{
    ast::{
        AnalyzedAssignExpr, AnalyzedBlock, AnalyzedCastExpr, AnalyzedExpression, AnalyzedForStmt,
        AnalyzedFunctionDefinition, AnalyzedIfExpr, AnalyzedInfixExpr, AnalyzedLetStmt,
        AnalyzedLoopStmt, AnalyzedPrefixExpr, AnalyzedProgram, AnalyzedStatement,
        AnalyzedWhileStmt,
    },
    AssignOp, InfixOp, PrefixOp, Type,
};

use crate::{
    instruction::{Condition, Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register},
    utils::{Block, DataObj, DataObjType, Function, Loop, Size, Variable, VariableValue},
};

pub struct Compiler {
    /// Exported functions
    pub(crate) exports: Vec<String>,
    /// Sections / basic blocks which contain instructions.
    pub(crate) blocks: Vec<Block>,
    /// Points to the current section which is inserted to
    pub(crate) curr_block: usize,
    /// Data section for storing global variables.
    pub(crate) data_section: Vec<DataObj>,
    /// Read-only data section for storing constant values.
    pub(crate) rodata_section: Vec<DataObj>,
    /// Holds metadata about the current function
    pub(crate) curr_fn: Option<Function>,
    /// Holds metadata about the current loop
    pub(crate) curr_loop: Option<Loop>,
    /// Saves the scopes. The last element is the most recent scope.
    pub(crate) scopes: Vec<HashMap<String, Option<Variable>>>,
    /// Holds the global variables of the program
    pub(crate) globals: HashMap<String, Option<Variable>>,
    /// Specifies all registers which are currently in use and may not be overwritten.
    pub(crate) used_registers: Vec<Register>,
}

const GLOBALS_INIT_LABEL: &str = "globals_init";

impl Compiler {
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

    pub fn compile(&mut self, ast: AnalyzedProgram) -> String {
        self.declare_main_fn(ast.main_fn, ast.globals);

        for func in ast.functions.into_iter().filter(|f| f.used) {
            self.function_declaration(func)
        }

        let mut output = String::new();

        output += ".section .text\n";

        // string generation
        output += &self
            .exports
            .iter()
            .map(|e| format!(".global {e}\n"))
            .collect::<String>();

        // block generation
        output += &self
            .blocks
            .iter()
            .map(|d| d.to_string())
            .collect::<String>();

        // .data generation
        if !self.data_section.is_empty() {
            output += &format!(
                "\n.section .data\n{}",
                self.data_section
                    .iter()
                    .map(|d| format!("\n{d}\n"))
                    .collect::<String>()
            );
        }

        // .rodata generation
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

    fn declare_main_fn(&mut self, node: AnalyzedBlock, globals: Vec<AnalyzedLetStmt>) {
        let _start = "_start";
        self.append_block(_start);
        self.exports.push(_start.into());

        let prologue_label = self.gen_label("main_prologue");

        if !globals.is_empty() {
            self.insert_jmp(GLOBALS_INIT_LABEL.to_string());
            self.append_block(GLOBALS_INIT_LABEL);
            self.insert_at(GLOBALS_INIT_LABEL);
            for var in globals {
                self.declare_global(var.name.to_string(), var.expr)
            }
        }
        self.insert_jmp(prologue_label.clone());

        self.blocks.push(Block {
            label: prologue_label.clone(),
            instructions: vec![],
        });

        let body = self.append_block("body");
        let epilogue_label = self.gen_label("epilogue");

        self.curr_fn = Some(Function {
            stack_allocs: 0,
            body_label: body.clone(),
            epilogue_label: epilogue_label.clone(),
        });

        self.push_scope();
        self.insert_at(&body);
        self.function_body(node);
        self.insert_jmp(epilogue_label.clone());
        self.pop_scope();

        // align frame size to 16 bytes
        Self::align(&mut self.curr_fn_mut().stack_allocs, 16);
        // make prologue
        self.prologue(&prologue_label, vec![]);

        // exit with code 0
        self.blocks.push(Block {
            label: epilogue_label.clone(),
            instructions: vec![],
        });
        self.insert_at(&epilogue_label);
        self.insert(Instruction::Li(IntRegister::A0, 0));
        self.insert(Instruction::Call("exit".into()));
    }

    fn declare_global(&mut self, label: String, value: AnalyzedExpression) {
        // initialize global value at the start of the program
        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment(format!("let {label} (global)")));

        let type_ = value.result_type();
        let res_reg = self
            .expression(value)
            .expect("cannot use unit value in globals");

        let data = match type_ {
            Type::Int => {
                self.insert(Instruction::Sd(
                    res_reg.into(),
                    Pointer::Label(label.clone()),
                ));
                DataObjType::Dword(0)
            }
            Type::Bool | Type::Char => {
                self.insert(Instruction::Sb(
                    res_reg.into(),
                    Pointer::Label(label.clone()),
                ));
                DataObjType::Byte(0)
            }
            Type::Float => {
                self.insert(Instruction::Fsd(
                    res_reg.into(),
                    Pointer::Label(label.clone()),
                ));
                DataObjType::Float(0.0)
            }
            _ => unreachable!("other types cannot be used in globals"),
        };

        let var = Variable {
            type_,
            value: VariableValue::Pointer(Pointer::Label(label.clone())),
        };
        self.globals.insert(label.clone(), Some(var));
        self.data_section.push(DataObj { label, data });
    }

    fn function_declaration(&mut self, node: AnalyzedFunctionDefinition) {
        // append block for the function
        let block_label = format!("main..{}", node.name);
        self.append_block(&block_label);
        self.insert_at(&block_label);

        let prologue_label = self.gen_label("prologue");
        let body_label = self.gen_label("body");
        let epilogue_label = self.gen_label("epilogue");

        self.curr_fn = Some(Function {
            stack_allocs: 0,
            body_label: body_label.clone(),
            epilogue_label: epilogue_label.clone(),
        });

        self.push_scope();

        let mut int_cnt = 0;
        let mut float_cnt = 0;
        let mut mem_offset = 0;

        let mut param_store_instructions = vec![
            #[cfg(debug_assertions)]
            Instruction::Comment("save params on stack".to_string()),
        ];

        for param in node.params {
            match param.type_ {
                Type::Int | Type::Char | Type::Bool => {
                    match IntRegister::nth_param(int_cnt) {
                        Some(reg) => {
                            let size = Size::from(param.type_);
                            Self::align(&mut self.curr_fn_mut().stack_allocs, size.byte_count());
                            self.curr_fn_mut().stack_allocs += size.byte_count();
                            let offset = -self.curr_fn().stack_allocs as i64 - 16;

                            match size {
                                Size::Byte => param_store_instructions.push(Instruction::Sb(
                                    reg,
                                    Pointer::Stack(IntRegister::Fp, offset),
                                )),
                                Size::Dword => param_store_instructions.push(Instruction::Sd(
                                    reg,
                                    Pointer::Stack(IntRegister::Fp, offset),
                                )),
                            }

                            self.scope_mut().insert(
                                param.name.to_string(),
                                Some(Variable {
                                    type_: param.type_,
                                    value: VariableValue::Pointer(Pointer::Stack(
                                        IntRegister::Fp,
                                        offset,
                                    )),
                                }),
                            );
                        }
                        None => {
                            self.scope_mut().insert(
                                param.name.to_string(),
                                Some(Variable {
                                    type_: param.type_,
                                    value: VariableValue::Pointer(Pointer::Stack(
                                        IntRegister::Fp,
                                        mem_offset,
                                    )),
                                }),
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

                            param_store_instructions.push(Instruction::Fsd(
                                reg,
                                Pointer::Stack(IntRegister::Fp, offset),
                            ));
                            self.scope_mut().insert(
                                param.name.to_string(),
                                Some(Variable {
                                    type_: param.type_,
                                    value: VariableValue::Pointer(Pointer::Stack(
                                        IntRegister::Fp,
                                        offset,
                                    )),
                                }),
                            );
                        }
                        None => {
                            self.scope_mut().insert(
                                param.name.to_string(),
                                Some(Variable {
                                    type_: param.type_,
                                    value: VariableValue::Pointer(Pointer::Stack(
                                        IntRegister::Fp,
                                        mem_offset,
                                    )),
                                }),
                            );
                            mem_offset += 8;
                        }
                    }
                    float_cnt += 1;
                }
                Type::Unit | Type::Never => {
                    // ignore these types
                    self.scope_mut().insert(param.name.to_string(), None);
                }
                Type::Unknown => unreachable!("analyzer would have failed"),
            }
        }

        // jump into the function prologue
        self.insert_jmp(prologue_label.to_string());

        // add the prologue block
        self.blocks.push(Block {
            label: prologue_label.clone(),
            instructions: vec![],
        });

        // generate function body
        self.blocks.push(Block {
            label: body_label.clone(),
            instructions: vec![],
        });
        self.insert_at(&body_label);
        self.function_body(node.block);
        self.insert_jmp(epilogue_label.clone());
        self.pop_scope();

        // align frame size to 16 bytes
        Self::align(&mut self.curr_fn_mut().stack_allocs, 16);
        // generate prologue
        self.prologue(&prologue_label, param_store_instructions);

        // generate epilogue
        self.blocks.push(Block {
            label: epilogue_label,
            instructions: vec![],
        });

        // generate epilogue
        self.epilogue()
    }

    fn function_body(&mut self, node: AnalyzedBlock) {
        // compile each statement
        for stmt in node.stmts {
            self.statement(stmt);
        }

        // place the result of the optional expression in the return value register(s)
        if let Some(expr) = node.expr {
            let res_reg = self.expression(expr);
            match res_reg {
                Some(Register::Int(IntRegister::A0))
                | Some(Register::Float(FloatRegister::Fa0)) => {} // already in target register
                Some(Register::Int(reg)) => {
                    self.insert(Instruction::Mov(IntRegister::A0, reg));
                }
                Some(Register::Float(reg)) => {
                    self.insert(Instruction::Fmv(FloatRegister::Fa0, reg));
                }
                None => {} // do nothing with unit values
            }
        }
    }

    fn block(&mut self, node: AnalyzedBlock) -> Option<Register> {
        // push a new scope
        self.push_scope();

        for stmt in node.stmts {
            self.statement(stmt)
        }

        // return expression register if there is an expr
        let res = match node.expr {
            Some(expr) => self.expression(expr),
            None => None,
        };

        // pop the scope again
        self.pop_scope();

        res
    }

    fn statement(&mut self, node: AnalyzedStatement) {
        match node {
            AnalyzedStatement::Let(node) => self.let_statement(node),
            AnalyzedStatement::Return(node) => {
                // if there is an expression, compile it
                if let Some(expr) = node {
                    match self.expression(expr) {
                        Some(Register::Int(reg)) => {
                            self.insert(Instruction::Mov(IntRegister::A0, reg))
                        }
                        Some(Register::Float(reg)) => {
                            self.insert(Instruction::Fmv(FloatRegister::Fa0, reg))
                        }
                        None => {}
                    }
                }
                self.insert_jmp(self.curr_fn().epilogue_label.clone());
            }
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

    fn loop_stmt(&mut self, node: AnalyzedLoopStmt) {
        let loop_head = self.append_block("loop_head");
        self.insert_jmp(loop_head.clone());
        self.insert_at(&loop_head);
        let after_loop_label = self.gen_label("after_loop");
        self.curr_loop = Some(Loop {
            loop_head: loop_head.clone(),
            after_loop: after_loop_label.clone(),
        });
        self.block(node.block);
        self.insert_jmp(loop_head);
        self.blocks.push(Block {
            label: after_loop_label.clone(),
            instructions: vec![],
        });
        self.insert_at(&after_loop_label);
    }

    fn while_stmt(&mut self, node: AnalyzedWhileStmt) {
        let while_loop_head = self.append_block("while_head");
        let after_loop_label = self.gen_label("after_while");
        self.insert_jmp(while_loop_head.clone());
        self.insert_at(&while_loop_head);

        // compile the condition
        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment("while condition".to_string()));
        let expr_res = self.expression(node.cond).expect("cond is always a bool");
        self.insert(Instruction::BrCond(
            Condition::Eq,
            IntRegister::Zero,
            expr_res.into(),
            after_loop_label.clone(),
        ));

        self.curr_loop = Some(Loop {
            loop_head: while_loop_head.clone(),
            after_loop: after_loop_label.clone(),
        });

        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment("while body".to_string()));

        self.block(node.block);
        self.insert_jmp(while_loop_head);
        self.blocks.push(Block {
            label: after_loop_label.clone(),
            instructions: vec![],
        });
        self.insert_at(&after_loop_label);
    }

    fn for_stmt(&mut self, node: AnalyzedForStmt) {
        let for_head = self.append_block("for_head");
        let after_loop = self.append_block("after_for");

        // compile the initialization expression
        let init_type = node.initializer.result_type();
        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment(format!(
            "loop initializer: {}",
            node.ident
        )));
        let res = self.expression(node.initializer).map(|reg| {
            self.use_reg(reg);
            (
                Variable {
                    type_: init_type,
                    value: VariableValue::Register(reg),
                },
                reg,
            )
        });
        self.push_scope();
        self.scope_mut()
            .insert(node.ident.to_string(), res.as_ref().map(|r| r.0.clone()));

        // compile the condition
        self.insert_at(&for_head);
        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment("loop condition".to_string()));
        let expr_res = self.expression(node.cond).expect("cond is always a bool");
        self.insert(Instruction::BrCond(
            Condition::Eq,
            IntRegister::Zero,
            expr_res.into(),
            after_loop.clone(),
        ));

        self.curr_loop = Some(Loop {
            loop_head: for_head.clone(),
            after_loop: after_loop.clone(),
        });

        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment("loop body".to_string()));

        // compile the block
        for stmt in node.block.stmts {
            self.statement(stmt)
        }

        if let Some(expr) = node.block.expr {
            self.expression(expr);
        }

        // compile the update expression
        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment("loop update".to_string()));
        self.expression(node.update);

        // jump back to the loop start
        self.insert_jmp(for_head);

        self.insert_at(&after_loop);

        self.pop_scope();
        if let Some(reg) = res.map(|r| r.1) {
            self.release_reg(reg)
        }
    }

    /// Allocates the variable on the stack.
    /// Also increments the `additional_stack_space` of the current function
    fn let_statement(&mut self, node: AnalyzedLetStmt) {
        let type_ = node.expr.result_type();

        let size = Size::from(type_).byte_count();
        Self::align(&mut self.curr_fn_mut().stack_allocs, size);
        self.curr_fn_mut().stack_allocs += size as i64;
        let offset = -self.curr_fn().stack_allocs as i64 - 16;

        #[cfg(debug_assertions)]
        self.insert(Instruction::Comment(format!("let {}", node.name)));
        let value_reg = self.expression(node.expr);

        match value_reg {
            Some(Register::Int(reg)) => self.insert(Instruction::Sd(
                reg,
                Pointer::Stack(IntRegister::Fp, offset),
            )),
            Some(Register::Float(reg)) => self.insert(Instruction::Fsd(
                reg,
                Pointer::Stack(IntRegister::Fp, offset),
            )),
            None => {
                // insert a dummy variable into the HashMap
                self.scopes
                    .last_mut()
                    .expect("there must be a scope")
                    .insert(node.name.to_string(), None);
            }
        }

        // insert variable into current scope
        let var = Variable {
            type_,
            value: VariableValue::Pointer(Pointer::Stack(IntRegister::Fp, offset)),
        };

        self.scopes
            .last_mut()
            .expect("there must be a scope")
            .insert(node.name.to_string(), Some(var));
    }

    pub(crate) fn expression(&mut self, node: AnalyzedExpression) -> Option<Register> {
        match node {
            AnalyzedExpression::Block(node) => self.block(*node),
            AnalyzedExpression::If(node) => self.if_expr(*node),
            AnalyzedExpression::Int(value) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Li(dest_reg, value));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Float(value) => {
                let dest_reg = self.alloc_freg();

                // if there is already a float constant with this label, use it
                // otherwise, create a new float constant and use its label
                let float_value_label = match self
                    .rodata_section
                    .iter()
                    .find(|o| o.data == DataObjType::Float(value))
                {
                    Some(obj) => obj.label.clone(),
                    None => {
                        let label = format!("float_constant_{}", self.rodata_section.len());
                        self.rodata_section.push(DataObj {
                            label: label.clone(),
                            data: DataObjType::Float(value),
                        });
                        label
                    }
                };

                // load value from float constant into `dest_reg`
                self.insert(Instruction::Fld(
                    dest_reg,
                    Pointer::Label(float_value_label),
                ));

                Some(Register::Float(dest_reg))
            }
            AnalyzedExpression::Bool(value) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Li(dest_reg, value as i64));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Char(value) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Li(dest_reg, value as i64));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Ident(ident) => {
                let var = match self.resolve_name(ident.ident).clone() {
                    Some(var) => var,
                    None => return None,
                };
                self.load_value_from_variable(var)
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
        }
    }

    fn load_value_from_variable(&mut self, var: Variable) -> Option<Register> {
        match var.value {
            VariableValue::Pointer(ptr) => match var.type_ {
                Type::Bool | Type::Char => {
                    let dest_reg = self.alloc_ireg();
                    self.insert(Instruction::Lb(dest_reg, ptr));
                    Some(Register::Int(dest_reg))
                }
                Type::Int => {
                    let dest_reg = self.alloc_ireg();
                    self.insert(Instruction::Ld(dest_reg, ptr));
                    Some(Register::Int(dest_reg))
                }
                Type::Float => {
                    let dest_reg = self.alloc_freg();
                    self.insert(Instruction::Fld(dest_reg, ptr));
                    Some(Register::Float(dest_reg))
                }
                Type::Unit | Type::Never => None,
                Type::Unknown => unreachable!("analyzer would have failed"),
            },
            VariableValue::Register(reg) => Some(reg),
        }
    }

    fn assign_expr(&mut self, node: AnalyzedAssignExpr) {
        let rhs_type = node.expr.result_type();

        let assignee = self
            .resolve_name(node.assignee)
            .as_ref()
            .expect("filtered above")
            .clone();

        let rhs_reg = match node.op {
            AssignOp::Basic => match self.expression(node.expr) {
                Some(reg) => reg,
                None => return,
            },
            _ => {
                // load value from the lhs
                let lhs = self
                    .load_value_from_variable(assignee.clone())
                    .expect("filtered above");
                self.use_reg(lhs);

                // compile the rhs
                let Some(rhs) = self.expression(node.expr) else {return };
                self.use_reg(rhs);

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
                _ => unreachable!("other types cannot exist in an assignment"),
            },
            VariableValue::Register(dest) => match rhs_type {
                Type::Int | Type::Bool | Type::Char => {
                    self.insert(Instruction::Mov((dest).into(), rhs_reg.into()))
                }
                Type::Float => todo!(),
                _ => unreachable!("other types cannot exist in an assignment"),
            },
        }
    }

    fn if_expr(&mut self, node: AnalyzedIfExpr) -> Option<Register> {
        // save the bool condition in this register
        let cond_reg = self
            .expression(node.cond)
            .expect("cond is not unit / never");

        // will later hold the result of this expr
        let res_reg = match node.result_type {
            Type::Float => Some(self.alloc_freg().to_reg()),
            Type::Int | Type::Bool | Type::Char => Some(self.alloc_ireg().to_reg()),
            _ => None,
        };

        let then_block = self.append_block("then");
        let merge_block = self.append_block("merge");

        self.insert(Instruction::BrCond(
            Condition::Ne,
            cond_reg.into(),
            IntRegister::Zero,
            then_block.clone(),
        ));

        if let Some(else_block) = node.else_block {
            let else_block_label = self.append_block("else");
            self.insert_jmp(else_block_label.clone());
            self.insert_at(&else_block_label);
            let else_reg = self.block(else_block);

            // if the block returns a register other than res, move the block register into res
            match (res_reg, else_reg) {
                (Some(Register::Int(res)), Some(Register::Int(else_reg))) if res != else_reg => {
                    self.insert(Instruction::Mov(res, else_reg));
                }
                (Some(Register::Float(res)), Some(Register::Float(else_reg)))
                    if res != else_reg =>
                {
                    self.insert(Instruction::Fmv(res, else_reg));
                }
                _ => {}
            }
        }

        self.insert_jmp(merge_block.clone());

        self.insert_at(&then_block);
        let then_reg = self.block(node.then_block);

        // if the block returns a register other than res, move the block register into res
        match (res_reg, then_reg) {
            (Some(Register::Int(res)), Some(Register::Int(then_reg))) if res != then_reg => {
                self.insert(Instruction::Mov(res, then_reg));
            }
            (Some(Register::Float(res)), Some(Register::Float(then_reg))) if res != then_reg => {
                self.insert(Instruction::Fmv(res, then_reg));
            }
            _ => {}
        }

        self.insert_jmp(merge_block.clone());
        self.insert_at(&merge_block);

        res_reg
    }

    fn infix_expr(&mut self, node: AnalyzedInfixExpr) -> Option<Register> {
        let type_ = node.lhs.result_type();

        // mark the lhs register as used
        let lhs_reg = self.expression(node.lhs)?;
        self.use_reg(lhs_reg);

        // mark the rhs register as used
        let rhs_reg = self.expression(node.rhs)?;
        self.use_reg(rhs_reg);

        let res = self.infix_helper(lhs_reg, rhs_reg, node.op, type_);

        // release the usage block of the operands
        self.release_reg(lhs_reg);
        self.release_reg(rhs_reg);

        Some(res)
    }

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
            (
                // even if not all ops are allowed for char and bool, the analyzer would not accept
                // illegal programs, threfore this is ok.
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
                let dest_regi = self.alloc_ireg();
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
            _ => unreachable!("the analyzer does not allow other combinations"),
        }
    }

    fn prefix_expr(&mut self, node: AnalyzedPrefixExpr) -> Option<Register> {
        let lhs_type = node.expr.result_type();
        let lhs_reg = self.expression(node.expr)?;

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
            (Type::Bool, PrefixOp::Not) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Seqz(dest_reg, lhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            _ => unreachable!("other types cannot occur in infix expressions"),
        }
    }

    fn cast_expr(&mut self, node: AnalyzedCastExpr) -> Option<Register> {
        let lhs_type = node.expr.result_type();
        let lhs_reg = self.expression(node.expr)?;

        // block the use of the lhs temporarily
        self.use_reg(lhs_reg);

        let res = match (lhs_type, node.type_) {
            (lhs, rhs) if lhs == rhs => lhs_reg,
            (Type::Bool, Type::Int) | (Type::Bool, Type::Char) | (Type::Char, Type::Int) => lhs_reg,
            (Type::Int, Type::Float) => {
                let dest_reg = self.alloc_freg();
                self.insert(Instruction::CastIntToFloat(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Int, Type::Bool) | (Type::Char, Type::Bool) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Snez(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Int, Type::Char) => self
                .__rush_internal_cast_int_to_char(lhs_reg.into())
                .to_reg(),
            (Type::Char, Type::Float) | (Type::Bool, Type::Float) => {
                let dest_reg = self.alloc_freg();
                self.insert(Instruction::CastByteToFloat(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Float, Type::Int) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::CastFloatToInt(dest_reg, lhs_reg.into()));
                dest_reg.to_reg()
            }
            (Type::Float, Type::Char) => self
                .__rush_internal_cast_float_to_char(lhs_reg.into())
                .to_reg(),
            (Type::Float, Type::Bool) => {
                // get a .rodata label which holds a float zero
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

                // compare the float to 0.0
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
            _ => unreachable!("cannot use this combination in a typecast"),
        };

        self.release_reg(lhs_reg);
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        process::{Command, Stdio},
        time::Instant,
    };

    use super::*;

    #[test]
    fn test_compiler() {
        let path = "./test.rush";
        let code = fs::read_to_string(path).unwrap();
        let (ast, _) = rush_analyzer::analyze(&code, path).unwrap();
        let start = Instant::now();
        let mut compiler = Compiler::new();
        let out = compiler.compile(ast);
        fs::write("test.s", out).unwrap();
        println!("compile: {:?}", start.elapsed());

        Command::new("riscv64-linux-gnu-gcc")
            .args([
                "-mno-relax",
                "-nostdlib",
                "-static",
                "-g",
                "test.s",
                "-L",
                "corelib",
                "-lcore",
                "-o",
                "test",
            ])
            .arg("-static")
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .output()
            .unwrap();
    }
}

#![allow(dead_code)] // TODO: remove this attribute

use std::collections::{HashMap, VecDeque};

use rush_analyzer::{
    ast::{
        AnalyzedBlock, AnalyzedCastExpr, AnalyzedExpression, AnalyzedFunctionDefinition,
        AnalyzedIfExpr, AnalyzedInfixExpr, AnalyzedLetStmt, AnalyzedProgram, AnalyzedStatement,
    },
    InfixOp, Type,
};

use crate::{
    instruction::{Condition, FldType, Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register},
    utils::{Block, DataObj, DataObjType, Function, Variable},
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
    /// Saves the scopes. The last element is the most recent scope.
    pub(crate) scopes: Vec<HashMap<String, Option<Variable>>>,
    /// Specifies all registers which are currently in use and may not be overwritten.
    pub(crate) used_registers: Vec<IntRegister>,
    /// Specifies all float registers which are currently in use and may not be overwritten.
    pub(crate) used_float_registers: Vec<FloatRegister>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            blocks: vec![],
            exports: vec![],
            curr_block: 0,
            data_section: vec![],
            rodata_section: vec![],
            scopes: vec![],
            curr_fn: None,
            used_registers: vec![],
            used_float_registers: vec![],
        }
    }

    pub fn compile(&mut self, ast: AnalyzedProgram) -> String {
        for _var in ast.globals {
            // TODO: add zero initializer, init variables in
        }

        self.declare_main_fn(ast.main_fn);

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

    fn declare_main_fn(&mut self, node: AnalyzedBlock) {
        let prologue = self.append_block("prologue".to_string());

        // append block for the function
        let _start = "_start";
        self.append_block(_start.into());
        self.exports.push(_start.into());

        let body = self.append_block("body".to_string());

        self.insert_at(_start);
        self.insert(Instruction::Jmp(prologue.clone()));

        let epilogue_label = self.gen_label("epilogue".to_string());

        self.curr_fn = Some(Function {
            stack_allocs: 0,
            body_label: body.clone(),
            epilogue_label: epilogue_label.clone(),
        });

        self.push_scope();
        self.insert_at(&body);
        self.function_body(node);
        self.insert(Instruction::Jmp(epilogue_label.clone()));
        self.pop_scope();

        // make prologue
        self.prologue(&prologue);

        // exit with code 0
        self.blocks.push(Block {
            label: epilogue_label.clone(),
            instructions: VecDeque::new(),
        });
        self.insert_at(&epilogue_label);
        self.insert(Instruction::Li(IntRegister::A0, 0));
        self.insert(Instruction::Call("exit".into()));
    }

    fn declare_global(&mut self, _ident: String, _value: AnalyzedExpression) {
        todo!()
    }

    fn function_declaration(&mut self, node: AnalyzedFunctionDefinition) {
        // append block for the function
        let block_label = format!(".{}", node.name);
        self.append_block(block_label.clone());
        self.insert_at(&block_label);

        let prologue_label = self.gen_label("prologue".to_string());
        let body_label = self.gen_label("body".to_string());
        let epilogue_label = self.gen_label("epilogue".to_string());

        self.curr_fn = Some(Function {
            stack_allocs: 0,
            body_label: body_label.clone(),
            epilogue_label: epilogue_label.clone(),
        });

        self.push_scope();

        // TODO: implement args

        for (idx, param) in node.params.iter().enumerate() {
            match param.type_ {
                Type::Int => {
                    // TODO: account for mixed types
                    let reg = IntRegister::nth_param(idx)
                        .expect("argument spilling is not yet implemented")
                        .to_reg();
                    self.scope_mut()
                        .insert(param.name.to_string(), Some(Variable::Register(reg)));
                }
                Type::Char | Type::Bool => {}
                Type::Float => {}
                Type::Unit | Type::Never => {} // ignore these types
                Type::Unknown => unreachable!("analyzer would have failed"),
            }
        }

        // jump into the function prologue
        self.insert(Instruction::Jmp(prologue_label.to_string()));

        // add the prologue block
        self.blocks.push(Block {
            label: prologue_label.clone(),
            instructions: VecDeque::new(),
        });

        // generate function body
        self.blocks.push(Block {
            label: body_label.clone(),
            instructions: VecDeque::new(),
        });
        self.insert_at(&body_label);
        self.function_body(node.block);
        self.insert(Instruction::Jmp(epilogue_label.clone()));
        self.pop_scope();

        // generate prologue
        self.prologue(&prologue_label);

        // generate epilogue
        self.blocks.push(Block {
            label: epilogue_label,
            instructions: VecDeque::new(),
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
                    self.insert(Instruction::Fmov(FloatRegister::Fa0, reg));
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
            AnalyzedStatement::Return(_) => todo!(),
            AnalyzedStatement::Loop(_) => todo!(),
            AnalyzedStatement::While(_) => todo!(),
            AnalyzedStatement::For(_) => todo!(),
            AnalyzedStatement::Break => todo!(),
            AnalyzedStatement::Continue => todo!(),
            AnalyzedStatement::Expr(node) => {
                self.expression(node);
            }
        }
    }

    /// Allocates the variable on the stack.
    /// Also increments the `additional_stack_space` of the current function
    fn let_statement(&mut self, node: AnalyzedLetStmt) {
        let value_reg = self.expression(node.expr);

        // store the value of the expr on the stack
        self.insert(Instruction::Comment(format!("let {}", node.name)));

        let stack_allocs = self.curr_fn().stack_allocs as i64;

        match value_reg {
            Some(Register::Int(reg)) => self.insert(Instruction::Sd(
                reg,
                Pointer::Stack(IntRegister::Fp, -(self.curr_fn().stack_allocs as i64)),
            )),
            Some(Register::Float(reg)) => self.insert(Instruction::Fsd(FldType::Stack(
                reg,
                IntRegister::Fp,
                -(self.curr_fn().stack_allocs as i64),
            ))),
            None => {
                // insert a dummy variable into the HashMap
                self.scopes
                    .last_mut()
                    .expect("there must be a scope")
                    .insert(node.name.to_string(), None);
            }
        }

        // insert variable into current scope
        let variable = Variable::Pointer(Pointer::Stack(IntRegister::Fp, -stack_allocs));
        self.scopes
            .last_mut()
            .expect("there must be a scope")
            .insert(node.name.to_string(), Some(variable));

        // TODO: is 8 a good value?
        self.curr_fn_mut().stack_allocs += 8;
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
                // add a float constant with the value
                let label = format!("float_constant_{}", self.rodata_section.len());
                self.rodata_section.push(DataObj {
                    label: label.clone(),
                    data: DataObjType::Float(value),
                });
                // load value from float constant into `dest_reg`
                self.insert(Instruction::Fld(FldType::Label(
                    dest_reg,
                    label,
                    self.alloc_ireg(),
                )));
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
            AnalyzedExpression::Ident(node) => {
                let var = self.resolve_name(node.ident);

                match var {
                    Some(Variable::Pointer(ptr)) => match node.result_type {
                        Type::Bool | Type::Char => {
                            let dest_reg = self.alloc_ireg();
                            self.insert(Instruction::Lb(dest_reg, ptr.clone()));
                            Some(Register::Int(dest_reg))
                        }
                        Type::Int => {
                            let dest_reg = self.alloc_ireg();
                            self.insert(Instruction::Ld(dest_reg, ptr.clone()));
                            Some(Register::Int(dest_reg))
                        }
                        Type::Float => {
                            let dest_reg = self.alloc_freg();
                            match ptr {
                                Pointer::Stack(reg, offset) => {
                                    self.insert(Instruction::Fld(FldType::Stack(
                                        dest_reg, *reg, *offset,
                                    )));
                                }
                                Pointer::Label(_) => todo!(),
                            }
                            Some(Register::Float(dest_reg))
                        }
                        Type::Unit | Type::Never => None,
                        Type::Unknown => unreachable!("analyzer would have failed"),
                    },
                    Some(Variable::Register(reg)) => Some(*reg),
                    None => None,
                }
            }
            AnalyzedExpression::Prefix(_) => todo!(),
            AnalyzedExpression::Infix(node) => self.infix_expr(*node),
            AnalyzedExpression::Assign(_) => todo!(),
            AnalyzedExpression::Call(node) => self.call_expr(*node),
            AnalyzedExpression::Cast(node) => self.cast_expr(*node),
            AnalyzedExpression::Grouped(node) => self.expression(*node),
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

        let then_block = self.append_block("then".to_string());
        let merge_block = self.append_block("merge".to_string());

        self.insert(Instruction::BrCond(
            Condition::Ne,
            cond_reg.into(),
            IntRegister::Zero,
            then_block.clone(),
        ));

        if let Some(else_block) = node.else_block {
            let else_block_label = self.append_block("else".to_string());
            self.insert(Instruction::Jmp(else_block_label.clone()));
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
                    self.insert(Instruction::Fmov(res, else_reg));
                }
                _ => {}
            }

            self.insert(Instruction::Jmp(merge_block.clone()));
        }

        self.insert_at(&then_block);
        let then_reg = self.block(node.then_block);

        // if the block returns a register other than res, move the block register into res
        match (res_reg, then_reg) {
            (Some(Register::Int(res)), Some(Register::Int(then_reg))) if res != then_reg => {
                self.insert(Instruction::Mov(res, then_reg));
            }
            (Some(Register::Float(res)), Some(Register::Float(then_reg))) if res != then_reg => {
                self.insert(Instruction::Fmov(res, then_reg));
            }
            _ => {}
        }

        self.insert(Instruction::Jmp(merge_block.clone()));
        self.insert_at(&merge_block);

        res_reg
    }

    fn infix_expr(&mut self, node: AnalyzedInfixExpr) -> Option<Register> {
        let type_ = node.lhs.result_type();

        let lhs_reg = self.expression(node.lhs)?;
        // mark the lhs register as used
        self.use_reg(lhs_reg);

        let rhs_reg = self.expression(node.rhs)?;
        // mark the rhs register as used
        self.use_reg(rhs_reg);

        let res = match (type_, node.op) {
            (Type::Int, InfixOp::Plus) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Add(dest_reg, lhs_reg.into(), rhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Int, InfixOp::Minus) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Sub(dest_reg, lhs_reg.into(), rhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Int, InfixOp::Mul) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Mul(dest_reg, lhs_reg.into(), rhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Int, InfixOp::Div) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Div(dest_reg, lhs_reg.into(), rhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Int, InfixOp::Rem) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Rem(dest_reg, lhs_reg.into(), rhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Int, InfixOp::Pow) => todo!("figure out calls first"),
            (Type::Float, InfixOp::Plus) => {
                let dest_reg = self.alloc_freg();
                self.insert(Instruction::Fadd(dest_reg, lhs_reg.into(), rhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Float, InfixOp::Minus) => {
                let dest_reg = self.alloc_freg();
                self.insert(Instruction::Fsub(dest_reg, lhs_reg.into(), rhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Float, InfixOp::Mul) => {
                let dest_reg = self.alloc_freg();
                self.insert(Instruction::Fmul(dest_reg, lhs_reg.into(), rhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Float, InfixOp::Div) => {
                let dest_reg = self.alloc_freg();
                self.insert(Instruction::Fdiv(dest_reg, lhs_reg.into(), rhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            _ => todo!("implement these cases"), // TODO: implement this
        };

        // release the usage block of the operands
        self.release_reg(lhs_reg);
        self.release_reg(rhs_reg);

        res
    }

    fn cast_expr(&mut self, node: AnalyzedCastExpr) -> Option<Register> {
        let lhs_type = node.expr.result_type();
        let lhs_reg = self.expression(node.expr)?;

        match (lhs_type, node.type_) {
            (Type::Float, Type::Int) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::CastFloatToInt(dest_reg, lhs_reg.into()));
                Some(dest_reg.to_reg())
            }
            (Type::Bool, Type::Int) | (Type::Char, Type::Int) => Some(lhs_reg),
            _ => todo!(),
        }
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

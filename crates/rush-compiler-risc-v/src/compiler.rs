#![allow(dead_code)] // TODO: remove this attribute

use std::collections::HashMap;

use rush_analyzer::ast::{
    AnalyzedBlock, AnalyzedExpression, AnalyzedFunctionDefinition, AnalyzedLetStmt,
    AnalyzedProgram, AnalyzedStatement,
};

use crate::{
    instruction::{FldType, Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register},
};

pub struct Compiler {
    /// Sections / basic blocks which contain instructions.
    pub(crate) blocks: Vec<Block>,
    /// Points to the current section which is inserted to
    pub(crate) curr_block: usize,
    /// Data section for storing global variables.
    pub(crate) data_section: Vec<DataObj>,
    /// Read-only data section for storing constant values.
    pub(crate) rodata_section: Vec<DataObj>,
    /// Holds metadata about the current function
    pub(crate) curr_fn: Function,
    /// Saves the scopes. The last element is the most recent scope.
    pub(crate) scopes: Vec<HashMap<String, i64 /* sp offset */>>,
    /// Specifies all registers which are currently in use and may not be overwritten.
    pub(crate) used_registers: Vec<IntRegister>,
    /// Specifies all float registers which are currently in use and may not be overwritten.
    pub(crate) used_float_registers: Vec<FloatRegister>,
}

pub(crate) struct Block {
    pub(crate) label: String,
    pub(crate) instructions: Vec<Instruction>,
}

pub(crate) struct Function {
    /// Specifies how many bytes of stack space need to be allocated for the current function.
    /// Include the necessary allocation for `ra` or `fp`
    pub(crate) fp: usize,
}

pub(crate) enum DataObj {
    Float(f64),
    Dword(i64),
    Byte(i64),
}

impl Compiler {
    fn position_at_end(&mut self, label: &str) {
        self.curr_block = self
            .blocks
            .iter()
            .position(|s| s.label == label)
            .expect("cannot insert at invalid label: {label}")
    }

    pub fn new() -> Self {
        Self {
            blocks: vec![],
            curr_block: 0,
            data_section: vec![],
            rodata_section: vec![],
            scopes: vec![],
            curr_fn: Function { fp: 0 },
            used_registers: vec![],
            used_float_registers: vec![],
        }
    }

    pub fn compile(&mut self, ast: AnalyzedProgram) -> String {
        for _var in ast.globals {
            // TODO: add zero initializer, init variables in
            //self.declare_global(var.name.into())
        }

        for func in ast.functions {
            self.function_declaration(func)
        }

        todo!()
    }

    fn declare_global(&mut self, _ident: String, _value: AnalyzedExpression) {
        todo!()
    }

    fn function_declaration(&mut self, node: AnalyzedFunctionDefinition) {
        // append block for the function
        let block_label = format!(".{}", node.name);
        self.append_block(block_label.clone());
        self.position_at_end(&block_label);

        // compile each statement
        for stmt in node.block.stmts {
            self.statement(stmt);
        }

        // place the result of the optional expression in the return value register(s)
        if let Some(expr) = node.block.expr {
            let res_reg = self.expression(expr);
            match res_reg {
                Register::Int(IntRegister::A0) | Register::Float(FloatRegister::Fa0) => {} // already in target register
                Register::Int(reg) => {
                    self.insert(Instruction::Mov(IntRegister::A0, reg));
                }
                Register::Float(reg) => {
                    self.insert(Instruction::Fmov(FloatRegister::Fa0, reg));
                }
            }
        }
    }

    fn block(&mut self, node: AnalyzedBlock) {
        // push a new scope
        self.push_scope();

        for stmt in node.stmts {
            self.statement(stmt)
        }

        // pop the scope again
        self.pop_scope()
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
        match value_reg {
            Register::Int(reg) => self.insert(Instruction::Sd(
                reg,
                Pointer::Stack(IntRegister::Fp, self.curr_fn.fp as i64),
            )),
            Register::Float(reg) => self.insert(Instruction::Fsd(FldType::Stack(
                reg,
                IntRegister::T0,
                self.curr_fn.fp as i64,
            ))),
        }

        // insert variable into current scope
        self.scopes
            .last_mut()
            .expect("there must be a scope")
            .insert(node.name.to_string(), self.curr_fn.fp as i64);

        // increment frame pointer
        self.curr_fn.fp += 8;
    }

    fn expression(&mut self, node: AnalyzedExpression) -> Register {
        match node {
            AnalyzedExpression::Block(_) => todo!(),
            AnalyzedExpression::If(_) => todo!(),
            AnalyzedExpression::Int(_) => todo!(),
            AnalyzedExpression::Float(_) => todo!(),
            AnalyzedExpression::Bool(_) => todo!(),
            AnalyzedExpression::Char(_) => todo!(),
            AnalyzedExpression::Ident(_) => todo!(),
            AnalyzedExpression::Prefix(_) => todo!(),
            AnalyzedExpression::Infix(_) => todo!(),
            AnalyzedExpression::Assign(_) => todo!(),
            AnalyzedExpression::Call(_) => todo!(),
            AnalyzedExpression::Cast(_) => todo!(),
            AnalyzedExpression::Grouped(_) => todo!(),
        }
    }
}

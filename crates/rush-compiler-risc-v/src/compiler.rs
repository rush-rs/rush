use std::collections::{HashMap, HashSet};

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

    pub(crate) curr_fn: Function,

    pub(crate) scopes: Vec<HashMap<String, i64 /* sp offset */>>,
    /// Specifies all registers which are currently in use and may not be overwritten.
    pub(crate) used_registers: HashSet<IntRegister>,
    /// Specifies all float registers which are currently in use and may not be overwritten.
    pub(crate) used_float_registers: HashSet<FloatRegister>,
}

pub(crate) struct Block {
    pub(crate) label: String,
    pub(crate) instructions: Vec<Instruction>,
}

pub(crate) struct Function {
    /// Specifies how many bytes of stack space need to be allocated for the current function.
    /// Include the nessecary allocation for `ra` or `fp`
    pub(crate) stack_space: usize,
}

pub(crate) enum DataObj {
    Float(f64),
    Dword(i64),
    Byte(i64),
}

impl Compiler {
    fn position_at_end(&mut self, label: String) {
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
            curr_fn: Function { stack_space: 0 },
            used_registers: HashSet::new(),
            used_float_registers: HashSet::new(),
        }
    }

    pub fn compile(&mut self, ast: AnalyzedProgram) -> String {
        for var in ast.globals {
            // TODO: add zero initializer, init variables in
            //self.declare_global(var.name.into())
        }

        for func in ast.functions {
            self.function_declaration(func)
        }

        todo!()
    }

    fn declare_global(&mut self, ident: String, value: AnalyzedExpression) {
        todo!()
    }

    fn function_declaration(&mut self, node: AnalyzedFunctionDefinition) {
        // append block for the function
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
            AnalyzedStatement::Expr(_) => todo!(),
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
                Pointer::Stack(IntRegister::Fp, self.curr_fn.stack_space as i64),
            )),
            Register::Float(reg) => self.insert(Instruction::Fsd(FldType::Stack(
                reg,
                IntRegister::T0,
                self.curr_fn.stack_space as i64,
            ))),
        }
        // TODO: insert into scope
        //
        // allocate stack space for the variable
        self.curr_fn.stack_space += 8;
    }

    fn expression(&mut self, node: AnalyzedExpression) -> Register {
        todo!()
    }
}

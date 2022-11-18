use std::collections::{HashMap, HashSet};

use rush_analyzer::ast::{AnalyzedExpression, AnalyzedFunctionDefinition, AnalyzedProgram};

use crate::{
    instruction::Instruction,
    register::{FloatRegister, Register},
};

pub struct Compiler {
    /// Sections / basic blocks which contain instructions.
    pub(crate) blocks: Vec<Block>,
    /// Points to the current section which is inserted to
    curr_block: usize,

    /// Data section for storing global variables.
    pub(crate) data_section: Vec<DataObj>,
    /// Read-only data section for storing constant values.
    pub(crate) rodata_section: Vec<DataObj>,

    pub(crate) curr_fn: Option<Function>,
    /// Specifies all registers which are currently in use and may not be overwritten.
    pub(crate) used_registers: HashSet<Register>,
    /// Specifies all float registers which are currently in use and may not be overwritten.
    pub(crate) used_float_registers: HashSet<FloatRegister>,
}

pub(crate) struct Block {
    label: String,
    instructions: Vec<Instruction>,
}

pub(crate) enum DataObj {
    Float(f64),
    Dword(i64),
    Byte(i64),
}

pub(crate) struct Function {
    pub(crate) vars: Vec<HashMap<String, i64 /* sp offset */>>,
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
            curr_fn: None,
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

    fn function_declaration(&mut self, node: AnalyzedFunctionDefinition) {}
}

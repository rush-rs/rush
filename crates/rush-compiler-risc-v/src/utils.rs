use std::collections::HashMap;

use crate::{
    compiler::{Block, Compiler},
    instruction::Instruction,
    register::{FloatRegister, IntRegister},
};

impl Compiler {
    pub(crate) fn alloc_reg(&mut self) -> IntRegister {
        todo!()
    }

    pub(crate) fn alloc_freg(&mut self) -> FloatRegister {
        todo!()
    }

    pub(crate) fn append_block(&mut self, label: String) -> usize {
        self.blocks.push(Block {
            label,
            instructions: vec![],
        });
        self.blocks.len() - 1
    }

    #[inline]
    pub(crate) fn insert(&mut self, instruction: Instruction) {
        self.blocks[self.curr_block].instructions.push(instruction);
    }

    #[inline]
    pub(crate) fn push_scope(&mut self) {
        self.scopes.push(HashMap::new())
    }

    #[inline]
    pub(crate) fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

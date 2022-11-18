#![allow(dead_code)] // TODO: remove this attribute

use std::collections::HashMap;

use crate::{
    compiler::{Block, Compiler},
    instruction::Instruction,
    register::{FloatRegister, IntRegister, FLOAT_REGISTERS, INT_REGISTERS},
};

impl Compiler {
    /// Allocates (and returns) the next unused, general purpose int register
    pub(crate) fn alloc_ireg(&self) -> IntRegister {
        for reg in INT_REGISTERS {
            if !self.used_registers.contains(reg) {
                return *reg;
            }
        }
        unreachable!("out of registers!")
    }

    /// Allocates (and returns) the next unused, general purpose float register
    pub(crate) fn alloc_freg(&self) -> FloatRegister {
        for reg in FLOAT_REGISTERS {
            if !self.used_float_registers.contains(reg) {
                return *reg;
            }
        }
        unreachable!("out of float registers!")
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

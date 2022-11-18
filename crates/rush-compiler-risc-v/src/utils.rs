#![allow(dead_code)] // TODO: remove this attribute

use std::collections::{HashMap, VecDeque};

use crate::{
    compiler::{Block, Compiler},
    instruction::Instruction,
    register::{FloatRegister, IntRegister, Register, FLOAT_REGISTERS, INT_REGISTERS},
};

impl Compiler {
    pub(crate) fn scope_mut(&mut self) -> &mut HashMap<String, Option<i64>> {
        self.scopes.last_mut().expect("always called from a scope")
    }

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

    /// Marks a register as used
    pub(crate) fn use_reg(&mut self, reg: Register) {
        match reg {
            Register::Int(reg) => self.used_registers.push(reg),
            Register::Float(reg) => self.used_float_registers.push(reg),
        }
    }

    /// Marks a register as unused
    pub(crate) fn release_reg(&mut self, reg: Register) {
        match reg {
            Register::Int(reg) => self.used_registers.retain(|r| *r != reg),
            Register::Float(reg) => self.used_float_registers.retain(|r| *r != reg),
        }
    }

    pub(crate) fn append_block(&mut self, label: String) -> usize {
        self.blocks.push(Block {
            label,
            instructions: VecDeque::new(),
        });
        self.blocks.len() - 1
    }

    #[inline]
    pub(crate) fn insert(&mut self, instruction: Instruction) {
        self.blocks[self.curr_block]
            .instructions
            .push_back(instruction);
    }

    #[inline]
    pub(crate) fn insert_start(&mut self, instruction: Instruction) {
        self.blocks[self.curr_block]
            .instructions
            .push_front(instruction);
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

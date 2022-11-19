#![allow(dead_code)] // TODO: remove this attribute

use std::{
    collections::{HashMap, VecDeque},
    fmt::Display,
};

use crate::{
    compiler::Compiler,
    instruction::{Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register, FLOAT_REGISTERS, INT_REGISTERS},
};

#[derive(Debug)]
pub(crate) enum Variable {
    Pointer(Pointer),
    Register(Register),
}

impl Compiler {
    pub(crate) fn scope_mut(&mut self) -> &mut HashMap<String, Option<Variable>> {
        self.scopes.last_mut().expect("always called from a scope")
    }

    pub(crate) fn curr_fn(&self) -> &Function {
        self.curr_fn.as_ref().expect("always called from functions")
    }

    pub(crate) fn curr_fn_mut(&mut self) -> &mut Function {
        self.curr_fn.as_mut().expect("always called from functions")
    }

    /// Allocates (and returns) the next unused, general purpose int register
    pub(crate) fn alloc_ireg(&self) -> IntRegister {
        for reg in INT_REGISTERS {
            if !self.used_registers.contains(&Register::Int(*reg)) {
                return *reg;
            }
        }
        unreachable!("out of registers!")
    }

    /// Allocates (and returns) the next unused, general purpose float register
    pub(crate) fn alloc_freg(&self) -> FloatRegister {
        for reg in FLOAT_REGISTERS {
            if !self.used_registers.contains(&Register::Float(*reg)) {
                return *reg;
            }
        }
        unreachable!("out of float registers!")
    }

    /// Returns whether a register is currently in use or not
    pub(crate) fn reg_used(&self, reg: &Register) -> bool {
        self.used_registers.contains(reg)
    }

    /// Marks a register as used
    pub(crate) fn use_reg(&mut self, reg: Register) {
        self.used_registers.push(reg)
    }

    /// Marks a register as unused.
    /// If the register appears twice in the vec, only remove one occurrence
    pub(crate) fn release_reg(&mut self, reg: Register) {
        self.used_registers.remove(
            self.used_registers
                .iter()
                .position(|r| *r == reg)
                .expect("register not in used_registers"),
        );
    }

    /// Appends a new basic block.
    /// If the specified label already exists, it is appended with a numeric suffix.
    /// The final label is then returned for later usage.
    pub(crate) fn append_block(&mut self, label: String) -> String {
        let label = self.gen_label(label);

        self.blocks.push(Block {
            label: label.clone(),
            instructions: VecDeque::new(),
        });

        label
    }

    pub(crate) fn gen_label(&self, label: String) -> String {
        let mut count = 1;
        let mut out = label.clone();
        while self.blocks.iter().any(|b| b.label == out) {
            out = format!("{label}_{}", count);
            count += 1;
        }
        out
    }

    /// Helper function for resolving identifier names.
    /// Searches the scopes first. If no match was found, the fitting global variable is returned.
    pub(crate) fn resolve_name(&self, name: &str) -> &Option<Variable> {
        for scope in self.scopes.iter().rev() {
            if let Some(variable) = scope.get(name) {
                return variable;
            }
        }
        dbg!(name, &self.scopes);
        unreachable!("every variable exists")
    }

    #[inline]
    pub(crate) fn insert(&mut self, instruction: Instruction) {
        self.blocks[self.curr_block]
            .instructions
            .push_back(instruction);
    }

    pub(crate) fn insert_at(&mut self, label: &str) {
        self.curr_block = self
            .blocks
            .iter()
            .position(|s| s.label == label)
            .expect("cannot insert at invalid label: {label}")
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

pub(crate) struct Block {
    pub(crate) label: String,
    pub(crate) instructions: VecDeque<Instruction>,
}

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n{}:\n{}",
            self.label,
            self.instructions
                .iter()
                .map(|i| format!("    {}\n", i.to_string().replace('\n', "\n    ")))
                .collect::<String>()
        )
    }
}

pub(crate) struct Function {
    /// Specifies how many bytes of stack space need to be allocated for the current function.
    /// Does not include the necessary allocations for `ra` or `fp`
    pub(crate) stack_allocs: usize,
    /// Holds the value of the label which contains the function body.
    pub(crate) body_label: String,
    /// Holds the value of the label which contains the epilogue of the current function.
    pub(crate) epilogue_label: String,
}

pub(crate) struct DataObj {
    pub(crate) label: String,
    pub(crate) data: DataObjType,
}

impl Display for DataObj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:\n    {}", self.label, self.data)
    }
}

pub(crate) enum DataObjType {
    Float(f64),
    Dword(i64),
    Byte(i64),
}

impl Display for DataObjType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(inner) => write!(
                f,
                ".dword {:#018x}  # = {inner}{zero}",
                inner.to_bits(),
                zero = if inner.fract() == 0.0 { ".0" } else { "" }
            ),
            Self::Dword(inner) => write!(f, ".dword {inner:#018x}  # = {inner}"),
            Self::Byte(inner) => write!(f, ".byte {inner:#04x}"),
        }
    }
}

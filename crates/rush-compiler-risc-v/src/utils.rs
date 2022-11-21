use std::{collections::HashMap, fmt::Display};

use rush_analyzer::Type;

use crate::{
    compiler::Compiler,
    instruction::{Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register, FLOAT_REGISTERS, INT_REGISTERS},
};

#[derive(Debug, Clone)]
pub(crate) struct Variable {
    pub(crate) type_: Type,
    pub(crate) value: VariableValue,
}

#[derive(Debug, Clone)]
pub(crate) enum VariableValue {
    Pointer(Pointer),
    Register(Register),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Size {
    Byte = 1,
    Dword = 8,
}

impl Size {
    pub(crate) fn byte_count(&self) -> i64 {
        *self as i64
    }
}

impl From<Type> for Size {
    fn from(src: Type) -> Self {
        match src {
            Type::Int | Type::Float => Size::Dword,
            Type::Bool | Type::Char => Size::Byte,
            Type::Unknown | Type::Never | Type::Unit => unreachable!("these types have no size"),
        }
    }
}

impl Compiler {
    pub(crate) fn align(ptr: &mut i64, size: i64) {
        if *ptr % size != 0 {
            *ptr += size - *ptr % size;
        }
    }

    pub(crate) fn scope_mut(&mut self) -> &mut HashMap<String, Option<Variable>> {
        self.scopes.last_mut().expect("always called from a scope")
    }

    pub(crate) fn curr_loop(&self) -> &Loop {
        self.curr_loop.as_ref().expect("always called from loops")
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
    pub(crate) fn append_block(&mut self, label: &str) -> String {
        let label = self.gen_label(label);

        self.blocks.push(Block {
            label: label.clone(),
            instructions: vec![],
        });

        label
    }

    pub(crate) fn gen_label(&self, label: &str) -> String {
        let mut count = 1;
        let mut out = label.to_string();
        while self.blocks.iter().any(|b| b.label == out) {
            out = format!("{label}_{}", count);
            count += 1;
        }
        out
    }

    /// Helper function for resolving identifier names.
    /// Searches the scopes first. If no match was found, the fitting global variable is returned.
    pub(crate) fn resolve_name(&self, name: &str) -> &Option<Variable> {
        // look for normal variables first
        for scope in self.scopes.iter().rev() {
            if let Some(variable) = scope.get(name) {
                return variable;
            }
        }
        // return reference to global variable
        self.globals.get(name).expect("every variable exists")
    }

    /// Inserts a jump at the current position.
    /// If the current block is already terminated, the insertion is omitted.
    pub(crate) fn insert_jmp(&mut self, label: String) {
        // check if the current block already contains a terminator
        let contains_jmp = self.blocks[self.curr_block]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Jmp(_)));

        // if there is no terminator, jump
        if !contains_jmp {
            self.insert(Instruction::Jmp(label))
        }
    }

    #[inline]
    pub(crate) fn insert(&mut self, instruction: Instruction) {
        self.blocks[self.curr_block].instructions.push(instruction);
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
    pub(crate) instructions: Vec<Instruction>,
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
    pub(crate) stack_allocs: i64,
    /// Holds the value of the label which contains the function body.
    pub(crate) body_label: String,
    /// Holds the value of the label which contains the epilogue of the current function.
    pub(crate) epilogue_label: String,
}

pub(crate) struct Loop {
    /// Specifies the `loop_head` label of the current loop
    pub(crate) loop_head: String,
    /// Specifies the `after_loop` label of the current loop
    pub(crate) after_loop: String,
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

#[derive(Debug, PartialEq)]
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

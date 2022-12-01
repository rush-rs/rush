use std::{collections::HashMap, fmt::Display};

use rush_analyzer::Type;

use crate::{
    compiler::Compiler,
    instruction::{Block, Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register, FLOAT_REGISTERS, INT_REGISTERS},
};

#[derive(Debug, Clone)]
pub(crate) struct Variable {
    pub(crate) type_: Type,
    pub(crate) value: VariableValue,
}

impl Variable {
    pub(crate) fn unit() -> Self {
        Self {
            type_: Type::Unit,
            value: VariableValue::Unit,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum VariableValue {
    Pointer(Pointer),
    Register(Register),
    Unit,
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

impl<'tree> Compiler<'tree> {
    // TODO: write documentation for this function
    pub(crate) fn align(ptr: &mut i64, size: i64) {
        if *ptr % size != 0 {
            *ptr += size - *ptr % size;
        }
    }

    pub(crate) fn get_offset(&mut self, size: Size) -> i64 {
        let size = size.byte_count();
        Self::align(&mut self.curr_fn_mut().stack_allocs, size);
        self.curr_fn_mut().stack_allocs += size as i64;
        -self.curr_fn().stack_allocs as i64 - 16
    }

    /// Saves a register to memory and returns its fp-offset.
    /// Used before function calls in order to save currently used registers to memory.
    pub(crate) fn spill_reg(&mut self, reg: Register, size: Size) -> i64 {
        let offset = self.get_offset(size);
        let comment = format!("{} byte spill: {reg}", size.byte_count());

        match reg {
            Register::Int(reg) => {
                let ptr = Pointer::Stack(IntRegister::Fp, offset);

                match size {
                    Size::Byte => self.insert_w_comment(Instruction::Sb(reg, ptr), comment),
                    Size::Dword => self.insert_w_comment(Instruction::Sd(reg, ptr), comment),
                }
            }
            Register::Float(reg) => self.insert(Instruction::Fsd(
                reg,
                Pointer::Stack(IntRegister::Fp, offset),
            )),
        };

        offset
    }

    pub(crate) fn restore_regs_after_call(
        &mut self,
        mut call_return_reg: Option<Register>,
        regs: Vec<(Register, i64, Size)>,
    ) -> Option<Register> {
        for (reg, offset, size) in regs {
            let comment = format!("{} byte reload: {reg}", size.byte_count());
            match reg {
                Register::Int(reg) => {
                    // in this case, restoring `a0` would destroy the call return value.
                    // therefore, the return value is copied into a new temporary register
                    if call_return_reg == Some(Register::Int(IntRegister::A0))
                        && reg == IntRegister::A0
                    {
                        let new_res_reg = self.alloc_ireg();
                        call_return_reg = Some(new_res_reg.to_reg());
                        // copy the return value into the new result value
                        self.insert(Instruction::Mov(new_res_reg, IntRegister::A0));
                    }

                    // perform different load operations depending on the size
                    match size {
                        Size::Byte => self.insert_w_comment(
                            Instruction::Lb(reg, Pointer::Stack(IntRegister::Fp, offset)),
                            comment,
                        ),
                        Size::Dword => self.insert_w_comment(
                            Instruction::Ld(reg, Pointer::Stack(IntRegister::Fp, offset)),
                            comment,
                        ),
                    };
                }
                Register::Float(reg) => {
                    // in this case, restoring `fa0` would destroy the call return value.
                    // therefore, the return value is copied into a new temporary register
                    if call_return_reg == Some(Register::Float(FloatRegister::Fa0))
                        && reg == FloatRegister::Fa0
                    {
                        let new_res_reg = self.alloc_freg();
                        call_return_reg = Some(new_res_reg.to_reg());
                        // copy the return value into the new result value
                        self.insert(Instruction::Fmv(new_res_reg, FloatRegister::Fa0));
                    }

                    self.insert_w_comment(
                        Instruction::Fld(reg, Pointer::Stack(IntRegister::Fp, offset)),
                        comment,
                    );
                }
            };
        }

        call_return_reg
    }

    /// Returns a mutable reference to the current scope.
    pub(crate) fn scope_mut(&mut self) -> &mut HashMap<&'tree str, Variable> {
        self.scopes.last_mut().expect("always called from a scope")
    }

    /// Returns a reference to the current loop being compiled.
    pub(crate) fn curr_loop(&self) -> &Loop {
        self.curr_loop.as_ref().expect("always called from loops")
    }

    /// Returns a reference to the current function being compiled.
    pub(crate) fn curr_fn(&self) -> &Function {
        self.curr_fn.as_ref().expect("always called from functions")
    }

    /// Returns a mutable reference to the current function being compiled.
    pub(crate) fn curr_fn_mut(&mut self) -> &mut Function {
        self.curr_fn.as_mut().expect("always called from functions")
    }

    /// Allocates and returns the next unused, general purpose int register.
    pub(crate) fn alloc_ireg(&self) -> IntRegister {
        for reg in INT_REGISTERS {
            if !self
                .used_registers
                .iter()
                .any(|(register, _)| register == &Register::Int(*reg))
            {
                return *reg;
            }
        }
        unreachable!("out of registers!")
    }

    /// Allocates and returns the next unused, general purpose float register.
    pub(crate) fn alloc_freg(&self) -> FloatRegister {
        for reg in FLOAT_REGISTERS {
            if !self
                .used_registers
                .iter()
                .any(|(register, _)| register == &Register::Float(*reg))
            {
                return *reg;
            }
        }
        unreachable!("out of float registers!")
    }

    /// Helper function which marks a register as used
    pub(crate) fn use_reg(&mut self, reg: Register, size: Size) {
        self.used_registers.push((reg, size))
    }

    /// Marks a register as unused.
    /// If the register appears twice in the vec, this only removes the first occurrence.
    pub(crate) fn release_reg(&mut self, reg: Register) {
        self.used_registers.remove(
            self.used_registers
                .iter()
                .position(|(r, _)| *r == reg)
                .expect("register not in used_registers"),
        );
    }

    /// Appends a new basic block.
    /// The initial label might be modified by the gen_label function.
    /// The final label is then returned for later usage.
    pub(crate) fn append_block(&mut self, label: &str) -> String {
        let label = self.gen_label(label);
        self.blocks.push(Block::new(label.clone()));
        label
    }

    /// Helperfunction for generating labels for basic blocks.
    /// If the specified label already exists, it gets a numeric suffix.
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
    /// Panics if the variable does not exist.
    pub(crate) fn resolve_name(&self, name: &str) -> &Variable {
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
        // only insert a jump if the current block is not already terminated
        if !self.curr_block().is_terminated {
            self.insert(Instruction::Jmp(label));
            self.curr_block_mut().is_terminated = true;
        }
    }

    #[inline]
    pub(crate) fn curr_block(&mut self) -> &Block {
        &self.blocks[self.curr_block]
    }

    #[inline]
    pub(crate) fn curr_block_mut(&mut self) -> &mut Block {
        &mut self.blocks[self.curr_block]
    }

    #[inline]
    /// Inserts an [`Instruction`] at the end of the current basic block.
    pub(crate) fn insert(&mut self, instruction: Instruction) {
        self.blocks[self.curr_block]
            .instructions
            .push((instruction, None));
    }

    /// Inserts an [`Instruction`] at the end of the current basic block.
    /// Also inserts the specified comment at the end of the instruction.
    pub(crate) fn insert_w_comment(&mut self, instruction: Instruction, comment: String) {
        self.blocks[self.curr_block]
            .instructions
            .push((instruction, Some(comment)));
    }

    /// Places the cursor at the end of the specified block.
    /// If the block label is invalid, the function panics.
    pub(crate) fn insert_at(&mut self, label: &str) {
        self.curr_block = self
            .blocks
            .iter()
            .position(|s| s.label == label)
            .expect("cannot insert at invalid label: {label}")
    }

    #[inline]
    /// Adds a new scope to the top of the stack.
    pub(crate) fn push_scope(&mut self) {
        self.scopes.push(HashMap::new())
    }

    #[inline]
    /// Removes the top of the scopes stack.
    pub(crate) fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

pub(crate) struct Function {
    /// Specifies how many bytes of stack memory are required by the current function.
    /// Does not include the necessary allocations for `ra` or `fp`
    pub(crate) stack_allocs: i64,
    /// Holds the value of the label which contains the epilogue of the current function.
    pub(crate) epilogue_label: String,
}

impl Function {
    /// Creates a new [`Function`].
    /// The `stack_allocs` is initialized as 0.
    pub(crate) fn new(epilogue_label: String) -> Self {
        Function {
            stack_allocs: 0,
            epilogue_label,
        }
    }
}

pub(crate) struct Loop {
    /// Specifies the `loop_head` label of the current loop.
    /// Used in the `continue` statement.
    pub(crate) loop_head: String,
    /// Specifies the `after_loop` label of the current loop.
    /// Used in the `break` statement.
    pub(crate) after_loop: String,
}

////////// Data Objects //////////

/// A [`DataObj`] is part of the `.data` and `.rodata` sections of the final program.
pub(crate) struct DataObj {
    /// The label / name of the data object.
    pub(crate) label: String,
    /// Holds the actual contents of the data object whilst saving type information.
    pub(crate) data: DataObjType,
}

impl Display for DataObj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:\n    {}", self.label, self.data)
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum DataObjType {
    /// Holds a float value of the `.dword` size.
    Float(f64),
    /// Holds an int value of the `.dword` size.
    Dword(i64),
    /// Holds 1 byte values like `char` and `bool`
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

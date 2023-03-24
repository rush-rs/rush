use std::{borrow::Cow, collections::HashMap, fmt::Display, rc::Rc};

use rush_analyzer::Type;

use crate::{
    compiler::Compiler,
    instruction::{Block, Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register, FLOAT_REGISTERS, INT_REGISTERS},
};

#[derive(Debug, Clone)]
pub(crate) struct Variable {
    pub(crate) type_: Type,
    /// This can be [`None`] if the type of the variable is `!` or `()`
    pub(crate) value: Option<Pointer>,
}

impl Variable {
    pub(crate) fn unit() -> Self {
        Self {
            type_: Type::Unit,
            value: None,
        }
    }
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
            Type::Bool(0) | Type::Char(0) => Size::Byte,
            Type::Int(_) | Type::Float(_) | Type::Bool(_) | Type::Char(_) => Size::Dword,
            Type::Unknown | Type::Never | Type::Unit => unreachable!("these types have no size"),
        }
    }
}

impl<'tree> Compiler<'tree> {
    // Aligns the specified pointer according to the specified size.
    // If the pointer is not a multiple of the specified size,
    // padding is added so that the pointer will be aligned.
    pub(crate) fn align(ptr: &mut i64, size: i64) {
        if *ptr % size != 0 {
            *ptr += size - *ptr % size;
        }
    }

    /// Helper function for allocation stack memory.
    /// Increments the current stack allocations and returns a new (aligned) index to be used.
    pub(crate) fn get_offset(&mut self, size: Size) -> i64 {
        let size = size.byte_count();
        Self::align(&mut self.curr_fn_mut().stack_allocs, size);
        self.curr_fn_mut().stack_allocs += size;
        -self.curr_fn().stack_allocs - 16
    }

    /// Saves a [`Register`] to memory and returns its fp-offset.
    /// Used before function calls in order to save currently used registers to memory.
    pub(crate) fn spill_reg(&mut self, reg: Register, size: Size) -> i64 {
        let offset = self.get_offset(size);
        let comment = format!("{} byte spill: {reg}", size.byte_count()).into();

        match reg {
            Register::Int(reg) => {
                let ptr = Pointer::Register(IntRegister::Fp, offset);

                match size {
                    Size::Byte => self.insert_with_comment(Instruction::Sb(reg, ptr), comment),
                    Size::Dword => self.insert_with_comment(Instruction::Sd(reg, ptr), comment),
                }
            }
            Register::Float(reg) => self.insert(Instruction::Fsd(
                reg,
                Pointer::Register(IntRegister::Fp, offset),
            )),
        };

        offset
    }

    /// Helper function which is invoked after a function call.
    /// Iterates through the given [`Register`]s in order to load their original value from the stack.
    /// Returns a possible register which contains the function's return value.
    pub(crate) fn restore_regs_after_call(
        &mut self,
        mut call_return_reg: Option<Register>,
        regs: Vec<(Register, i64, Size)>,
    ) -> Option<Register> {
        for (reg, offset, size) in regs {
            let comment = format!("{} byte reload: {reg}", size.byte_count()).into();
            match reg {
                Register::Int(reg) => {
                    // in this case, restoring `a0` would destroy the call return value.
                    // therefore, the return value is copied into a new temporary register
                    if call_return_reg == Some(Register::Int(IntRegister::A0))
                        && reg == IntRegister::A0
                    {
                        let new_res_reg = self.get_int_reg();
                        call_return_reg = Some(new_res_reg.to_reg());
                        // copy the return value into the new result value
                        self.insert(Instruction::Mv(new_res_reg, IntRegister::A0));
                    }

                    // perform different load operations depending on the size
                    match size {
                        Size::Byte => self.insert_with_comment(
                            Instruction::Lb(reg, Pointer::Register(IntRegister::Fp, offset)),
                            comment,
                        ),
                        Size::Dword => self.insert_with_comment(
                            Instruction::Ld(reg, Pointer::Register(IntRegister::Fp, offset)),
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
                        let new_res_reg = self.get_float_reg();
                        call_return_reg = Some(new_res_reg.to_reg());
                        // copy the return value into the new result value
                        self.insert(Instruction::Fmv(new_res_reg, FloatRegister::Fa0));
                    }

                    self.insert_with_comment(
                        Instruction::Fld(reg, Pointer::Register(IntRegister::Fp, offset)),
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
        self.loops
            .last()
            .as_ref()
            .expect("always called from loops")
    }

    /// Returns a reference to the current function being compiled.
    pub(crate) fn curr_fn(&self) -> &Function {
        self.curr_fn.as_ref().expect("always called from functions")
    }

    /// Returns a mutable reference to the current function being compiled.
    pub(crate) fn curr_fn_mut(&mut self) -> &mut Function {
        self.curr_fn.as_mut().expect("always called from functions")
    }

    /// Allocates and returns the next unused int register.
    /// Does not mark the register as used.
    pub(crate) fn get_int_reg(&self) -> IntRegister {
        for reg in INT_REGISTERS {
            if !self
                .used_registers
                .iter()
                .any(|(register, _)| register == &Register::Int(*reg))
            {
                return *reg;
            }
        }
        panic!("out of integer registers")
    }

    /// Allocates and returns the next unused float register.
    /// Does not mark the register as used.
    pub(crate) fn get_float_reg(&self) -> FloatRegister {
        for reg in FLOAT_REGISTERS {
            if !self
                .used_registers
                .iter()
                .any(|(register, _)| register == &Register::Float(*reg))
            {
                return *reg;
            }
        }
        panic!("out of float registers")
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
    pub(crate) fn append_block(&mut self, label: &'static str) -> Rc<str> {
        let label = self.gen_label(label);
        self.blocks.push(Block::new(Rc::clone(&label)));
        label
    }

    /// Helperfunction for generating labels for basic blocks.
    /// If the specified label already exists, it gets a numeric suffix.
    pub(crate) fn gen_label(&mut self, label: &'static str) -> Rc<str> {
        match self.label_count.get_mut(label) {
            Some(cnt) => {
                *cnt += 1;
                format!("{label}_{cnt}")
            }
            None => {
                self.label_count.insert(label, 0);
                format!("{label}_0")
            }
        }
        .into()
    }

    /// Helper function for resolving identifier names.
    /// Searches the scopes first. If no match was found, the matching global variable is returned.
    /// Panics if the variable does not exist.
    pub(crate) fn resolve_name(&self, name: &str) -> &Variable {
        // look for normal variables first
        for scope in self.scopes.iter().rev() {
            if let Some(variable) = scope.get(name) {
                return variable;
            }
        }
        // return reference to global variable
        self.globals
            .get(name)
            .expect("the analyzer guarantees valid variable references")
    }

    /// Loads the specified variable into a register.
    /// Decides which load operation is to be used as it depends on the data size.
    pub(crate) fn load_value_from_pointer(
        &mut self,
        ptr: Pointer,
        type_: Type,
        ident: &'tree str,
    ) -> Register {
        match type_ {
            Type::Bool(0) | Type::Char(0) => {
                let dest_reg = self.get_int_reg();
                self.insert_with_comment(Instruction::Lb(dest_reg, ptr), ident.into());
                Register::Int(dest_reg)
            }
            Type::Float(0) => {
                let dest_reg = self.get_float_reg();
                self.insert_with_comment(Instruction::Fld(dest_reg, ptr), ident.into());
                Register::Float(dest_reg)
            }
            Type::Int(_) | Type::Float(_) | Type::Bool(_) | Type::Char(_) => {
                let dest_reg = self.get_int_reg();
                self.insert_with_comment(Instruction::Ld(dest_reg, ptr), ident.into());
                Register::Int(dest_reg)
            }
            Type::Unit | Type::Never | Type::Unknown => {
                unreachable!("these values cannot be stored inside pointers")
            }
        }
    }

    /// Inserts a jump at the current position.
    /// If the current block is already terminated, the insertion is omitted.
    pub(crate) fn insert_jmp(&mut self, label: Rc<str>, comment: Option<Cow<'tree, str>>) {
        // only insert a jump if the current block is not already terminated
        if !self.curr_block().is_terminated {
            match comment {
                Some(comment) => self.insert_with_comment(Instruction::Jmp(label), comment),
                None => self.insert(Instruction::Jmp(label)),
            }
            self.curr_block_mut().is_terminated = true;
        }
    }

    #[inline]
    pub(crate) fn curr_block(&mut self) -> &Block {
        &self.blocks[self.curr_block]
    }

    #[inline]
    pub(crate) fn curr_block_mut(&mut self) -> &mut Block<'tree> {
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
    pub(crate) fn insert_with_comment(&mut self, instruction: Instruction, comment: Cow<'tree, str>) {
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
            .position(|s| *s.label == *label)
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
    pub(crate) epilogue_label: Rc<str>,
}

impl Function {
    /// Creates a new [`Function`].
    /// The `stack_allocs` is initialized as 0.
    pub(crate) fn new(epilogue_label: Rc<str>) -> Self {
        Function {
            stack_allocs: 0,
            epilogue_label,
        }
    }
}

pub(crate) struct Loop {
    /// Specifies the `loop_head` label of the current loop.
    /// Used in the `continue` statement.
    pub(crate) loop_head: Rc<str>,
    /// Specifies the `after_loop` label of the current loop.
    /// Used in the `break` statement.
    pub(crate) after_loop: Rc<str>,
}

impl Loop {
    pub(crate) fn new(loop_head: Rc<str>, after_loop: Rc<str>) -> Self {
        Self {
            loop_head,
            after_loop,
        }
    }
}

////////// Data Objects //////////

/// A [`DataObj`] is part of the `.data` and `.rodata` sections of the final program.
pub(crate) struct DataObj {
    /// The label / name of the data object.
    pub(crate) label: Rc<str>,
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
    Byte(u8),
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
            Self::Byte(inner) => write!(f, ".byte {inner:#04x}  # = {inner}"),
        }
    }
}

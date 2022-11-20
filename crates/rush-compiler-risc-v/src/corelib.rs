use crate::{
    compiler::Compiler,
    instruction::{Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register},
};

impl Compiler {
    /// Helper function for the `**` and `**=` operators.
    /// Because the RISC-V ISA does not support the pow instruction, the corelib is used.
    /// This function calls the `__rush_internal_pow_int` function in the rush corelib.
    pub(crate) fn __rush_internal_pow_int(
        &mut self,
        base: IntRegister,
        exponent: IntRegister,
    ) -> IntRegister {
        // before the function is called, all used registers must be pushed on the stack
        let mut regs_on_stack = vec![];

        // save all registers which are currently in use
        for reg in self.used_registers.clone() {
            let offset = -(self.curr_fn().stack_allocs as i64 + 8);
            match reg {
                Register::Int(reg) => {
                    let ptr = Pointer::Stack(IntRegister::Fp, offset);
                    self.insert(Instruction::Sd(reg, ptr));
                }
                Register::Float(reg) => self.insert(Instruction::Fsd(
                    reg,
                    Pointer::Stack(IntRegister::Fp, offset),
                )),
            };
            self.curr_fn_mut().stack_allocs += 8;
            regs_on_stack.push((reg, offset));
        }

        // prepare the arguments
        if base != IntRegister::A0 {
            self.insert(Instruction::Mov(IntRegister::A0, base));
        }
        if exponent != IntRegister::A1 {
            self.insert(Instruction::Mov(IntRegister::A1, exponent));
        }

        // perform the function call
        self.insert(Instruction::Call("__rush_internal_pow_int".to_string()));

        let mut res_reg = IntRegister::A0;

        // restore all saved registers
        for (reg, offset) in regs_on_stack {
            match reg {
                Register::Int(reg) => {
                    // in this case, restoring `a0` would destroy the call return value.
                    // therefore, the return value is copied into a new temporary register
                    if reg == IntRegister::A0 {
                        res_reg = self.alloc_ireg();
                        // copy the return value into the new result value
                        self.insert(Instruction::Mov(res_reg, IntRegister::A0));
                    }
                    self.insert(Instruction::Ld(
                        reg,
                        Pointer::Stack(IntRegister::Fp, offset),
                    ));
                }
                Register::Float(reg) => {
                    self.insert(Instruction::Fld(
                        reg,
                        Pointer::Stack(IntRegister::Fp, offset),
                    ));
                }
            };
        }

        res_reg
    }

    /// Calls the `__rush_internal_cast_int_to_char` function in the rush corelib.
    pub(crate) fn __rush_internal_cast_int_to_char(&mut self, src: IntRegister) -> IntRegister {
        // before the function is called, all used registers must be pushed on the stack
        let mut regs_on_stack = vec![];

        // save all registers which are currently in use
        for reg in self.used_registers.clone() {
            let offset = -(self.curr_fn().stack_allocs as i64 + 8);
            match reg {
                Register::Int(reg) => {
                    let ptr = Pointer::Stack(IntRegister::Fp, offset);
                    self.insert(Instruction::Sd(reg, ptr));
                }
                Register::Float(reg) => self.insert(Instruction::Fsd(
                    reg,
                    Pointer::Stack(IntRegister::Fp, offset),
                )),
            };
            self.curr_fn_mut().stack_allocs += 8;
            regs_on_stack.push((reg, offset));
        }

        // prepare the argument
        if src != IntRegister::A0 {
            self.insert(Instruction::Mov(IntRegister::A0, src));
        }

        // perform the function call
        self.insert(Instruction::Call(
            "__rush_internal_cast_int_to_char".to_string(),
        ));

        let mut res_reg = IntRegister::A0;

        // restore all saved registers
        for (reg, offset) in regs_on_stack {
            match reg {
                Register::Int(reg) => {
                    // in this case, restoring `a0` would destroy the call return value.
                    // therefore, the return value is copied into a new temporary register
                    if reg == IntRegister::A0 {
                        res_reg = self.alloc_ireg();
                        // copy the return value into the new result value
                        self.insert(Instruction::Mov(res_reg, IntRegister::A0));
                    }
                    self.insert(Instruction::Ld(
                        reg,
                        Pointer::Stack(IntRegister::Fp, offset),
                    ));
                }
                Register::Float(reg) => {
                    self.insert(Instruction::Fld(
                        reg,
                        Pointer::Stack(IntRegister::Fp, offset),
                    ));
                }
            };
        }

        res_reg
    }

    /// Calls the `__rush_internal_cast_float_to_char` function in the rush corelib.
    pub(crate) fn __rush_internal_cast_float_to_char(&mut self, src: FloatRegister) -> IntRegister {
        // before the function is called, all used registers must be pushed on the stack
        let mut regs_on_stack = vec![];

        // save all registers which are currently in use
        for reg in self.used_registers.clone() {
            let offset = -(self.curr_fn().stack_allocs as i64 + 8);
            match reg {
                Register::Int(reg) => {
                    let ptr = Pointer::Stack(IntRegister::Fp, offset);
                    self.insert(Instruction::Sd(reg, ptr));
                }
                Register::Float(reg) => self.insert(Instruction::Fsd(
                    reg,
                    Pointer::Stack(IntRegister::Fp, offset),
                )),
            };
            self.curr_fn_mut().stack_allocs += 8;
            regs_on_stack.push((reg, offset));
        }

        // prepare the argument
        if src != FloatRegister::Fa0 {
            self.insert(Instruction::Fmv(FloatRegister::Fa0, src));
        }

        // perform the function call
        self.insert(Instruction::Call(
            "__rush_internal_cast_float_to_char".to_string(),
        ));

        let mut res_reg = IntRegister::A0;

        // restore all saved registers
        for (reg, offset) in regs_on_stack {
            match reg {
                Register::Int(reg) => {
                    // in this case, restoring `a0` would destroy the call return value.
                    // therefore, the return value is copied into a new temporary register
                    if reg == IntRegister::A0 {
                        res_reg = self.alloc_ireg();
                        // copy the return value into the new result value
                        self.insert(Instruction::Mov(res_reg, IntRegister::A0));
                    }
                    self.insert(Instruction::Ld(
                        reg,
                        Pointer::Stack(IntRegister::Fp, offset),
                    ));
                }
                Register::Float(reg) => {
                    self.insert(Instruction::Fld(
                        reg,
                        Pointer::Stack(IntRegister::Fp, offset),
                    ));
                }
            };
        }

        res_reg
    }
}

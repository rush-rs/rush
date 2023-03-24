use crate::{
    compiler::Compiler,
    instruction::Instruction,
    register::{FloatRegister, IntRegister},
};

impl<'tree> Compiler<'tree> {
    /// Helper function for the `**` and `**=` operators.
    /// Because the RISC-V ISA does not support the pow instruction, the corelib is used.
    /// This function calls the `__rush_internal_pow_int` function in the rush corelib.
    pub(crate) fn __rush_internal_pow_int(
        &mut self,
        base: IntRegister,
        exponent: IntRegister,
    ) -> IntRegister {
        // before the function is called, all currently used registers are saved
        let regs_on_stack = self
            .used_registers
            .clone()
            .iter()
            .map(|(reg, size)| (*reg, self.spill_reg(*reg, *size), *size))
            .collect();

        // prepare the arguments
        if base != IntRegister::A0 {
            self.insert(Instruction::Mv(IntRegister::A0, base));
        }
        if exponent != IntRegister::A1 {
            self.insert(Instruction::Mv(IntRegister::A1, exponent));
        }

        // perform the function call
        self.insert(Instruction::Call("__rush_internal_pow_int".into()));

        // restore all saved registers
        self.restore_regs_after_call(Some(IntRegister::A0.to_reg()), regs_on_stack)
            .expect("is int")
            .into()
    }

    /// Calls the `__rush_internal_cast_int_to_char` function in the rush corelib.
    pub(crate) fn __rush_internal_cast_int_to_char(&mut self, src: IntRegister) -> IntRegister {
        // before the function is called, all currently used registers are saved
        let regs_on_stack = self
            .used_registers
            .clone()
            .iter()
            .map(|(reg, size)| (*reg, self.spill_reg(*reg, *size), *size))
            .collect();

        // prepare the argument
        if src != IntRegister::A0 {
            self.insert(Instruction::Mv(IntRegister::A0, src));
        }

        // perform the function call
        self.insert(Instruction::Call("__rush_internal_cast_int_to_char".into()));

        // restore all saved registers
        self.restore_regs_after_call(Some(IntRegister::A0.to_reg()), regs_on_stack)
            .expect("is char")
            .into()
    }

    /// Calls the `__rush_internal_cast_float_to_char` function in the rush corelib.
    pub(crate) fn __rush_internal_cast_float_to_char(&mut self, src: FloatRegister) -> IntRegister {
        // before the function is called, all currently used registers are saved
        let regs_on_stack = self
            .used_registers
            .clone()
            .iter()
            .map(|(reg, size)| (*reg, self.spill_reg(*reg, *size), *size))
            .collect();

        // prepare the argument
        if src != FloatRegister::Fa0 {
            self.insert(Instruction::Fmv(FloatRegister::Fa0, src));
        }

        // perform the function call
        self.insert(Instruction::Call(
            "__rush_internal_cast_float_to_char".into(),
        ));

        // restore all saved registers
        self.restore_regs_after_call(Some(IntRegister::A0.to_reg()), regs_on_stack)
            .expect("is char")
            .into()
    }
}

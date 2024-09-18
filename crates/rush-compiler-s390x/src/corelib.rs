use crate::{
    compiler::Compiler,
    instruction::Instruction,
    register::{FloatRegister, IntRegister, Register}, utils::Size,
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
        let regs_on_stack: Vec<(Register, i64, Size)> = self
            .used_registers
            .clone()
            .iter()
            .map(|(reg, size)| (*reg, self.spill_reg(*reg, *size), *size))
            .collect();

        dbg!(&regs_on_stack);

        // prepare the arguments
        // TODO: remove the hacky-ness and save r2 in between.
        if exponent != IntRegister::R3 {
            self.insert_with_comment(Instruction::Lgr(IntRegister::R3.into(), exponent.into()), "pow_int exponent".into());
        }

        if base != IntRegister::R2 {
            self.insert_with_comment(Instruction::Lgr(IntRegister::R2.into(), base.into()), "pow_int base".into());
        }


        // perform the function call
        self.insert_call("__rush_internal_pow_int".into());
        // self.insert(Instruction::Brasl());

        // restore all saved registers
        self.restore_regs_after_call(Some(IntRegister::R2.to_reg()), regs_on_stack)
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
        if src != IntRegister::R2 {
            self.insert_movi(IntRegister::R2, src, file!(), line!());
        }

        // perform the function call
        self.insert_call("__rush_internal_cast_int_to_char".into());

        // restore all saved registers
        self.restore_regs_after_call(Some(IntRegister::R2.to_reg()), regs_on_stack)
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
        if src != FloatRegister::F0 {
            self.insert_movf(FloatRegister::F0, src);
        }

        // perform the function call
        self.insert_call("__rush_internal_cast_float_to_char".into());

        // restore all saved registers
        self.restore_regs_after_call(Some(IntRegister::R2.to_reg()), regs_on_stack)
            .expect("is char")
            .into()
    }
}

use crate::{
    compiler::Compiler,
    instruction::{Instruction, Pointer},
    register::IntRegister,
};

impl Compiler {
    pub(crate) fn prologue(&mut self) {
        let prologue_label = self.curr_fn().prologue_label.clone();
        self.insert_at(&prologue_label);

        self.insert(Instruction::Addi(
            IntRegister::Sp,
            IntRegister::Sp,
            -(self.curr_fn().stack_allocs as i64),
        ));
        self.insert(Instruction::Sd(
            IntRegister::Fp,
            Pointer::Stack(IntRegister::Sp, 0),
        ));
        self.insert(Instruction::Sd(
            IntRegister::Ra,
            Pointer::Stack(IntRegister::Sp, 8),
        ));
        self.insert(Instruction::Addi(
            IntRegister::Fp,
            IntRegister::Sp,
            self.curr_fn().stack_allocs as i64,
        ));
        self.insert(Instruction::Jmp(self.curr_fn().body_label.clone()));
    }

    pub(crate) fn epilogue(&mut self) {
        let epilogue_label = self.curr_fn().epilogue_label.clone();
        self.insert_at(&epilogue_label);
        // restore fp
        self.insert(Instruction::Ld(
            IntRegister::Fp,
            Pointer::Stack(IntRegister::Sp, 0),
        ));
        // restore ra
        self.insert(Instruction::Ld(
            IntRegister::Ra,
            Pointer::Stack(IntRegister::Sp, 8),
        ));
        // deallocate stack space
        self.insert(Instruction::Addi(
            IntRegister::Sp,
            IntRegister::Sp,
            self.curr_fn().stack_allocs as i64,
        ));
        // return control back to caller
        self.insert(Instruction::Ret);
    }
}

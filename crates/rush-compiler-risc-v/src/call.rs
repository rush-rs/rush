use std::vec;

use rush_analyzer::{ast::AnalyzedCallExpr, Type};

use crate::{
    compiler::Compiler,
    instruction::{FldType, Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register},
};

impl Compiler {
    pub(crate) fn prologue(&mut self, label: &str) {
        self.insert_at(label);
        self.insert(Instruction::Addi(
            IntRegister::Sp,
            IntRegister::Sp,
            -(self.curr_fn().stack_allocs as i64 + 16/* 16 accounts for fp and ra */),
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
            self.curr_fn().stack_allocs as i64 + 16,
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
            self.curr_fn().stack_allocs as i64 + 16,
        ));
        // return control back to caller
        self.insert(Instruction::Ret);
    }

    /// TODO: add real implementation
    pub(crate) fn call_expr(&mut self, node: AnalyzedCallExpr) -> Option<Register> {
        let func_label = match node.func {
            "exit" => "exit".to_string(),
            func => format!(".{func}"),
        };

        let mut _float_cnt = 0;
        let mut int_cnt = 0;

        let mut param_regs = vec![];

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

                Register::Float(reg) => self.insert(Instruction::Fsd(FldType::Stack(
                    reg,
                    self.alloc_ireg(),
                    offset,
                ))),
            };
            self.curr_fn_mut().stack_allocs += 8;
            regs_on_stack.push((reg, offset));
        }

        // prepare arguments
        for arg in node
            .args
            .into_iter()
            .filter(|a| !matches!(a.result_type(), Type::Unit | Type::Never))
        {
            match arg.result_type() {
                Type::Int | Type::Bool | Type::Char => {
                    let curr_reg = IntRegister::nth_param(int_cnt)
                        .expect("spilling is not yet implemented")
                        .to_reg();

                    let res_reg = self
                        .expression(arg)
                        .expect("none variants filtered out above");

                    param_regs.push(curr_reg);
                    self.use_reg(curr_reg);

                    if curr_reg != res_reg {
                        dbg!(&self.used_registers);
                        let offset = -(self.curr_fn().stack_allocs as i64);
                        self.insert(Instruction::Sd(
                            IntRegister::A0,
                            Pointer::Stack(IntRegister::Fp, offset),
                        ));
                        self.curr_fn_mut().stack_allocs += 8;
                        self.insert(Instruction::Mov(curr_reg.into(), res_reg.into()));
                        //todo!("call: {}/{int_cnt}: register {curr_reg:?} is already in use, must save /load -> got {res_reg:?}", node.func)
                    }

                    int_cnt += 1;
                }
                Type::Float => {
                    todo!()
                }
                _ => unreachable!("either filtered out or impossible"),
            }
            // TODO: save args which are currently in use
        }

        // perform function call
        self.insert(Instruction::Call(func_label));

        let mut res_reg = match node.result_type {
            Type::Int | Type::Char | Type::Bool => Some(IntRegister::A0.to_reg()),
            Type::Float => Some(FloatRegister::Fa0.to_reg()),
            Type::Unit | Type::Never => None,
            Type::Unknown => unreachable!("analyzer would have failed"),
        };

        // restore all saved registers
        for (reg, offset) in regs_on_stack {
            println!("restored: {reg:?}");
            match reg {
                Register::Int(reg) => {
                    // in this case, restoring `a0` would destroy the call return value.
                    // therefore, the return value is copied into a new temporary register
                    if res_reg == Some(Register::Int(IntRegister::A0)) && reg == IntRegister::A0 {
                        let new_res_reg = self.alloc_ireg();
                        res_reg = Some(new_res_reg.to_reg());
                        // copy the return value into the new result value
                        self.insert(Instruction::Mov(new_res_reg, IntRegister::A0));
                    }
                    self.insert(Instruction::Ld(
                        reg,
                        Pointer::Stack(IntRegister::Fp, offset),
                    ));
                }
                Register::Float(reg) => self.insert(Instruction::Fld(FldType::Stack(
                    reg,
                    self.alloc_ireg(),
                    offset,
                ))),
            };
        }

        // free all blocked registers
        for reg in param_regs {
            self.release_reg(reg)
        }

        res_reg
    }
}

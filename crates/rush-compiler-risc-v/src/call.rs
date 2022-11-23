use std::vec;

use rush_analyzer::{ast::AnalyzedCallExpr, Type};

use crate::{
    compiler::Compiler,
    instruction::{Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register},
    utils::Size,
};

/// An offset of 16 accounts for `fp` and `ra`.
const BASE_STACK_ALLOCATIONS: i64 = 16;

impl Compiler {
    /// Returns the instructions of a function prologue.
    /// Automatically sets up any stack allocations and saves `ra` and `fp`.
    /// Must be invoked after the fn body since the stack frame size must be known at this point.
    pub(crate) fn prologue(&mut self) -> Vec<(Instruction, Option<String>)> {
        // align frame size to 16 bytes
        Self::align(&mut self.curr_fn_mut().stack_allocs, 16);

        vec![
            (Instruction::Comment("begin prologue".to_string()), None),
            // allocate stack spacec
            (
                Instruction::Addi(
                    IntRegister::Sp,
                    IntRegister::Sp,
                    -(self.curr_fn().stack_allocs as i64) - BASE_STACK_ALLOCATIONS,
                ),
                None,
            ),
            // save `fp` on the stack
            (
                Instruction::Sd(
                    IntRegister::Fp,
                    Pointer::Stack(IntRegister::Sp, self.curr_fn().stack_allocs + 8),
                ),
                None,
            ),
            // save `ra` on the stack
            (
                Instruction::Sd(
                    IntRegister::Ra,
                    Pointer::Stack(IntRegister::Sp, self.curr_fn().stack_allocs),
                ),
                None,
            ),
            // also offset the `fp` to start at the new frame
            (
                Instruction::Addi(
                    IntRegister::Fp,
                    IntRegister::Sp,
                    self.curr_fn().stack_allocs as i64 + BASE_STACK_ALLOCATIONS,
                ),
                None,
            ),
            (Instruction::Comment("end prologue".to_string()), None),
        ]
    }

    /// Inserts the instructions for a function epilogue
    /// at the end of the epilogue label of the current function .
    pub(crate) fn epilogue(&mut self) {
        let epilogue_label = self.curr_fn().epilogue_label.clone();
        self.insert_at(&epilogue_label);
        // restore `fp` from the stack
        self.insert(Instruction::Ld(
            IntRegister::Fp,
            Pointer::Stack(IntRegister::Sp, self.curr_fn().stack_allocs + 8),
        ));
        // restore `ra` from the stack
        self.insert(Instruction::Ld(
            IntRegister::Ra,
            Pointer::Stack(IntRegister::Sp, self.curr_fn().stack_allocs),
        ));
        // deallocate stack space
        self.insert(Instruction::Addi(
            IntRegister::Sp,
            IntRegister::Sp,
            self.curr_fn().stack_allocs as i64 + BASE_STACK_ALLOCATIONS,
        ));
        // return control back to caller
        self.insert(Instruction::Ret);
    }

    // TODO: document + refactor this function
    pub(crate) fn call_expr(&mut self, node: AnalyzedCallExpr) -> Option<Register> {
        let func_label = match node.func {
            "exit" => "exit".to_string(),
            func => format!("main..{func}"),
        };

        // before the function is called, all used registers must be saved
        let mut regs_on_stack = vec![];
        for (reg, size) in self.used_registers.clone() {
            let offset = self.spill_reg(reg, size);
            regs_on_stack.push((reg, offset, size));
        }

        let mut float_cnt = 0;
        let mut int_cnt = 0;

        // calculate the total byte size of spilled params
        let mut spill_param_size = 0;
        for arg in node
            .args
            .iter()
            .filter(|a| !matches!(a.result_type(), Type::Unit | Type::Never))
        {
            match arg.result_type() {
                Type::Int | Type::Bool | Type::Char => {
                    if IntRegister::nth_param(int_cnt).is_none() {
                        spill_param_size += 8;
                    }
                    int_cnt += 1;
                }
                Type::Float => {
                    if IntRegister::nth_param(float_cnt).is_none() {
                        spill_param_size += 8;
                    }

                    float_cnt += 1;
                }
                _ => unreachable!("either filtered out or unreachable"),
            }
        }

        int_cnt = 0;
        float_cnt = 0;

        let mut param_regs = vec![];

        // if there are caller saved registers, allocate space on the stack
        if spill_param_size > 0 {
            self.insert(Instruction::Addi(
                IntRegister::Sp,
                IntRegister::Sp,
                -spill_param_size,
            ));
        }

        let mut spill_count = 0;

        for arg in node
            .args
            .into_iter()
            .filter(|a| !matches!(a.result_type(), Type::Unit | Type::Never))
        {
            match arg.result_type() {
                Type::Int | Type::Bool | Type::Char => {
                    let type_ = arg.result_type();
                    let res_reg = self
                        .expression(arg)
                        .expect("none variants filtered out above");

                    match IntRegister::nth_param(int_cnt) {
                        Some(curr_reg) => {
                            param_regs.push(curr_reg.to_reg());
                            self.use_reg(curr_reg.to_reg(), Size::from(type_));

                            if curr_reg.to_reg() != res_reg {
                                let size = Size::from(type_).byte_count();
                                Self::align(&mut self.curr_fn_mut().stack_allocs, size);
                                self.curr_fn_mut().stack_allocs += size as i64;
                                let offset = -self.curr_fn().stack_allocs as i64 - 16;

                                match type_ {
                                    Type::Int => {
                                        self.insert(Instruction::Sd(
                                            curr_reg,
                                            Pointer::Stack(IntRegister::Fp, offset),
                                        ));
                                    }
                                    Type::Bool | Type::Char => {
                                        self.insert(Instruction::Sb(
                                            curr_reg,
                                            Pointer::Stack(IntRegister::Fp, offset),
                                        ));
                                    }
                                    _ => unreachable!(""),
                                }

                                self.insert(Instruction::Mov(curr_reg, res_reg.into()));
                            }
                        }
                        None => {
                            self.insert(Instruction::Sd(
                                res_reg.into(),
                                Pointer::Stack(IntRegister::Sp, spill_count * 8),
                            ));
                            spill_count += 1;
                        }
                    }

                    int_cnt += 1;
                }
                Type::Float => {
                    let type_ = arg.result_type();
                    let res_reg = self
                        .expression(arg)
                        .expect("none variants filtered out above");

                    match FloatRegister::nth_param(float_cnt) {
                        Some(curr_reg) => {
                            param_regs.push(curr_reg.to_reg());
                            self.use_reg(curr_reg.to_reg(), Size::from(type_));

                            if curr_reg.to_reg() != res_reg {
                                let size = Size::from(type_).byte_count();
                                Self::align(&mut self.curr_fn_mut().stack_allocs, size);
                                self.curr_fn_mut().stack_allocs += size as i64;
                                let offset = -self.curr_fn().stack_allocs as i64 - 16;

                                self.insert(Instruction::Fsd(
                                    curr_reg,
                                    Pointer::Stack(IntRegister::Fp, offset),
                                ));
                                self.insert(Instruction::Fmv(curr_reg, res_reg.into()));
                            }
                        }
                        None => {
                            self.insert(Instruction::Fsd(
                                res_reg.into(),
                                Pointer::Stack(IntRegister::Sp, spill_count * 8),
                            ));
                            spill_count += 1;
                        }
                    }
                    float_cnt += 1;
                }
                _ => unreachable!("either filtered out or impossible"),
            }
        }

        // perform function call
        self.insert(Instruction::Call(func_label));

        // if there were spilled params, deallocate stack space
        if spill_param_size > 0 {
            self.insert(Instruction::Addi(
                IntRegister::Sp,
                IntRegister::Sp,
                spill_param_size,
            ));
        }

        // free all blocked param registers
        for reg in param_regs {
            self.release_reg(reg)
        }

        let res_reg = match node.result_type {
            Type::Int | Type::Char | Type::Bool => Some(IntRegister::A0.to_reg()),
            Type::Float => Some(FloatRegister::Fa0.to_reg()),
            Type::Unit | Type::Never => None,
            Type::Unknown => unreachable!("analyzer would have failed"),
        };

        // restore all caller saved registers again
        self.restore_regs_after_call(res_reg, regs_on_stack)
    }
}

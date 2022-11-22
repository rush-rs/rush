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
    pub(crate) fn prologue(&mut self) -> Vec<Instruction> {
        // align frame size to 16 bytes
        Self::align(&mut self.curr_fn_mut().stack_allocs, 16);

        vec![
            #[cfg(debug_assertions)]
            Instruction::Comment("begin prologue".to_string()),
            // allocate stack spacec
            Instruction::Addi(
                IntRegister::Sp,
                IntRegister::Sp,
                -(self.curr_fn().stack_allocs as i64) - BASE_STACK_ALLOCATIONS,
            ),
            // save `fp` on the stack
            Instruction::Sd(
                IntRegister::Fp,
                Pointer::Stack(IntRegister::Sp, self.curr_fn().stack_allocs + 8),
            ),
            // save `ra` on the stack
            Instruction::Sd(
                IntRegister::Ra,
                Pointer::Stack(IntRegister::Sp, self.curr_fn().stack_allocs),
            ),
            // also offset the `fp` to start at the new frame
            Instruction::Addi(
                IntRegister::Fp,
                IntRegister::Sp,
                self.curr_fn().stack_allocs as i64 + BASE_STACK_ALLOCATIONS,
            ),
            #[cfg(debug_assertions)]
            Instruction::Comment("end prologue".to_string()),
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

        let mut float_cnt = 0;
        let mut int_cnt = 0;

        let mut param_regs = vec![];

        // before the function is called, all used registers must be pushed on the stack
        let mut regs_on_stack = vec![];

        // save all registers which are currently in use
        for reg in self.used_registers.clone() {
            let size = 8; // TODO: use size of the register instead if 8 bytes -> impl size for the
                          // used registers

            Self::align(&mut self.curr_fn_mut().stack_allocs, size);
            self.curr_fn_mut().stack_allocs += size as i64;
            let offset = -self.curr_fn().stack_allocs as i64 - 16;

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
            regs_on_stack.push((reg, offset));
        }

        // prepare arguments
        let mut sp_offset = 0;

        for arg in node
            .args
            .iter()
            .filter(|a| !matches!(a.result_type(), Type::Unit | Type::Never))
        {
            match arg.result_type() {
                Type::Int | Type::Bool | Type::Char => {
                    if IntRegister::nth_param(int_cnt).is_none() {
                        sp_offset += 8;
                    }
                    int_cnt += 1;
                }
                Type::Float => {
                    if IntRegister::nth_param(float_cnt).is_none() {
                        sp_offset += 8;
                    }

                    float_cnt += 1;
                }
                _ => unreachable!("either filtered out or unreachable"),
            }
        }

        int_cnt = 0;
        float_cnt = 0;

        if sp_offset > 0 {
            self.insert(Instruction::Addi(
                IntRegister::Sp,
                IntRegister::Sp,
                -sp_offset,
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
                            self.use_reg(curr_reg.to_reg());

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
                            self.use_reg(curr_reg.to_reg());

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

        if sp_offset > 0 {
            self.insert(Instruction::Addi(
                IntRegister::Sp,
                IntRegister::Sp,
                sp_offset,
            ));
        }

        let mut res_reg = match node.result_type {
            Type::Int | Type::Char | Type::Bool => Some(IntRegister::A0.to_reg()),
            Type::Float => Some(FloatRegister::Fa0.to_reg()),
            Type::Unit | Type::Never => None,
            Type::Unknown => unreachable!("analyzer would have failed"),
        };

        // restore all saved registers
        for (reg, offset) in regs_on_stack {
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
                Register::Float(reg) => {
                    // in this case, restoring `fa0` would destroy the call return value.
                    // therefore, the return value is copied into a new temporary register
                    if res_reg == Some(Register::Float(FloatRegister::Fa0))
                        && reg == FloatRegister::Fa0
                    {
                        let new_res_reg = self.alloc_freg();
                        res_reg = Some(new_res_reg.to_reg());
                        // copy the return value into the new result value
                        self.insert(Instruction::Fmv(new_res_reg, FloatRegister::Fa0));
                    }

                    self.insert(Instruction::Fld(
                        reg,
                        Pointer::Stack(IntRegister::Fp, offset),
                    ));
                }
            };
        }

        // free all blocked registers
        for reg in param_regs {
            self.release_reg(reg)
        }

        res_reg
    }
}

use std::{mem, vec};

use rush_analyzer::{ast::AnalyzedCallExpr, Type};

use crate::{
    compiler::Compiler,
    instruction::{Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register},
    utils::Size,
};

/// Specifies the default stack offset to be allocated.
/// An offset of 16 accounts for `fp` and `ra`.
const BASE_STACK_ALLOCATIONS: i64 = 16;

impl<'src> Compiler<'src> {
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
    pub(crate) fn call_expr(&mut self, node: &'src AnalyzedCallExpr) -> Option<Register> {
        // before the function is called, all currently used registers are saved
        let mut regs_on_stack = vec![];

        // TODO: differentiate between caller-saved and non caller-saved
        for (reg, size) in self.used_registers.clone() {
            let offset = self.spill_reg(reg, size);
            regs_on_stack.push((reg, offset, size));
        }

        // save the previous state of the used registers
        let used_regs_prev = mem::take(&mut self.used_registers);

        // specifies the place of the current register (relative to its type)
        // `a0` would be `int_cnt = 0`, `fa0` would be `float_cnt = 0`
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

        // reset counters for new iterations
        int_cnt = 0;
        float_cnt = 0;

        // will later contain all registers used as params (needed for releasing locks later)
        let mut param_regs = vec![];

        // if there are caller saved registers, allocate space on the stack
        if spill_param_size > 0 {
            self.insert(Instruction::Addi(
                IntRegister::Sp,
                IntRegister::Sp,
                -spill_param_size,
            ));
        }

        // specifies the count of the current register spill
        let mut spill_count = 0;

        for arg in &node.args {
            match arg.result_type() {
                Type::Unit | Type::Never => continue,
                Type::Int | Type::Bool | Type::Char => {
                    let type_ = arg.result_type();
                    let res_reg = self.expression(arg).expect("filtered out above");

                    match IntRegister::nth_param(int_cnt) {
                        Some(curr_reg) => {
                            param_regs.push(curr_reg.to_reg());
                            self.use_reg(curr_reg.to_reg(), Size::from(type_));

                            // TODO: try to remove this move
                            // if the reg from the expr is not the expected one, move it
                            if curr_reg.to_reg() != res_reg {
                                self.insert_w_comment(
                                    Instruction::Mov(curr_reg, res_reg.into()),
                                    format!("{res_reg} not expected {curr_reg}"),
                                )
                            }
                        }
                        None => {
                            // no more params: spilling required
                            self.insert_w_comment(
                                Instruction::Sd(
                                    res_reg.into(),
                                    Pointer::Stack(IntRegister::Sp, spill_count * 8),
                                ),
                                format!("{} byte param spill", Size::from(type_).byte_count()),
                            );
                            spill_count += 1;
                        }
                    }

                    int_cnt += 1;
                }
                Type::Float => {
                    let res_reg = self
                        .expression(arg)
                        .expect("none variants filtered out above");

                    match FloatRegister::nth_param(float_cnt) {
                        Some(curr_reg) => {
                            param_regs.push(curr_reg.to_reg());
                            self.use_reg(curr_reg.to_reg(), Size::Dword);

                            // if the reg from the expr is not the expected one, move it
                            if curr_reg.to_reg() != res_reg {
                                self.insert_w_comment(
                                    Instruction::Fmv(curr_reg, res_reg.into()),
                                    format!("{res_reg} not expected {curr_reg}"),
                                )
                            }
                        }
                        None => {
                            // no more params: spilling required
                            self.insert_w_comment(
                                Instruction::Fsd(
                                    res_reg.into(),
                                    Pointer::Stack(IntRegister::Sp, spill_count * 8),
                                ),
                                format!("{} byte param spill", Size::Dword.byte_count()),
                            );
                            spill_count += 1;
                        }
                    }
                    float_cnt += 1;
                }
                _ => unreachable!("either filtered out or impossible"),
            }
        }

        // perform function call
        let func_label = match node.func {
            "exit" => {
                // mark the current block as terminated (avoid future useless jumps)
                self.curr_block_mut().is_terminated = true;
                "exit".to_string()
            }
            func => format!("main..{func}"),
        };
        self.insert(Instruction::Call(func_label));

        // if there were spilled params, deallocate stack space
        if spill_param_size > 0 {
            self.insert(Instruction::Addi(
                IntRegister::Sp,
                IntRegister::Sp,
                spill_param_size,
            ));
        }

        // restore the old list of used registers
        self.used_registers = used_regs_prev;

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

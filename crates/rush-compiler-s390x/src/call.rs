use std::{borrow::Cow, mem, rc::Rc, vec};

use rush_analyzer::{ast::AnalyzedCallExpr, Type};

use crate::{
    compiler::Compiler,
    instruction::{Instruction, IntRegisterPointer},
    register::{FloatRegister, IntRegister, Register},
    utils::Size,
};

/// Specifies the default stack offset to be allocated.
/// An offset of 8 accounts for `GR15`.
pub(crate) const BASE_STACK_ALLOCATIONS: i64 = 8;

impl<'tree> Compiler<'tree> {
    /// Returns the instructions of a function prologue.
    /// Automatically sets up any stack allocations and saves `ra` and `fp`.
    /// Must be invoked after the fn body since the stack frame size must be known at this point.
    pub(crate) fn prologue(&mut self) -> Vec<(Instruction, Option<Cow<'static, str>>)> {
        // align frame size to 8 bytes
        Self::align(&mut self.curr_fn_mut().stack_allocs, 8);

        vec![
            (Instruction::Comment("begin prologue".into()), None),
            // allocate stack space
            (
                Instruction::Aghi(
                    IntRegister::R15,
                    (-self.curr_fn().stack_allocs - BASE_STACK_ALLOCATIONS)
                        .try_into()
                        .expect("offset too large"),
                ),
                Some("alloc frame".into()),
            ),
            // save `GR14` on the stack
            (
                Instruction::Store64(IntRegister::R14, IntRegisterPointer(IntRegister::R15, 0)),
                Some("save GR14".into()),
            ),
            (Instruction::Comment("end prologue".into()), None),
        ]
    }

    /// Inserts the instructions for a function epilogue.
    /// Places the instructions at the end of the epilogue label of the current function.
    pub(crate) fn epilogue(&mut self) {
        let epilogue_label = Rc::clone(&self.curr_fn().epilogue_label);
        self.insert_at(&epilogue_label);

        // // restore `GR14` from the stack
        self.insert_with_comment(
            Instruction::Load64(IntRegister::R14, IntRegisterPointer(IntRegister::R15, 0)),
            "restore GR14".into(),
        );

        // deallocate stack space
        self.insert_with_comment(
            Instruction::Aghi(
                IntRegister::R15,
                (self.curr_fn().stack_allocs + BASE_STACK_ALLOCATIONS)
                    .try_into()
                    .expect("offset too large"),
            ),
            "dealloc frame".into(),
        );

        // return control back to caller
        self.insert(Instruction::BranchRegister(IntRegister::R14));
    }

    /// Compiles an [`AnalyzedCallExpr`].
    /// Prior to calling the target function, all currently used registers are saved on the stack.
    /// After the call has been performed, all previously saved registers are restored from memory.
    pub(crate) fn call_expr(&mut self, node: AnalyzedCallExpr<'tree>) -> Option<Register> {
        // before the function is called, all currently used registers are saved
        let mut regs_on_stack: Vec<(Register, i64, Size)> = self
            .used_registers
            .clone()
            .iter()
            .map(|(reg, size)| (*reg, self.spill_reg(*reg, *size), *size))
            .collect();

        // save the previous state of the used registers
        let used_regs_prev = mem::take(&mut self.used_registers);

        // specifies the argument position of the specified register type
        // type dependent: (`a0` -> `int_cnt = 0`), (`fa0` -> `float_cnt = 0`)
        let mut float_cnt = 0;
        let mut int_cnt = 0;
        // calculate the total byte size of spilled params
        let mut spill_param_size = 0;
        // needed for freeing registers later
        let mut param_regs = vec![];
        // specifies the count of the current register spill
        let mut spill_cnt = 0;

        let gr13_offset =
            self.save_ireg_on_stack(IntRegister::R13, Some("save GR13 for params".into()));
        regs_on_stack.push((IntRegister::R13.into(), gr13_offset, Size::Double));

        self.insert_movi(IntRegister::R13, IntRegister::R15, file!(), line!());
        self.insert(Instruction::Aghi(
            IntRegister::R13,
            (self.curr_fn().stack_allocs + BASE_STACK_ALLOCATIONS)
                .try_into()
                .unwrap(),
        ));

        for arg in node.args {
            match arg.result_type() {
                Type::Unit | Type::Never | Type::Unknown => {
                    self.expression(arg);
                }
                Type::Float(0) => {
                    let res_reg = self.expression(arg).expect("type is float");

                    if let Some(reg) = FloatRegister::nth_param(float_cnt) {
                        param_regs.push(reg.to_reg());
                        self.use_reg(reg.to_reg(), Size::Double);
                    } else {
                        // no more param registers: spilling required
                        self.insert_with_comment(
                            Instruction::Std(
                                res_reg,
                                IntRegisterPointer(IntRegister::R13, spill_cnt * 8),
                            ),
                            format!("{} byte param spill", Size::Double.byte_count(),).into(),
                        );
                        spill_cnt += 1;
                        spill_param_size += 8;
                    }
                    float_cnt += 1;
                }
                Type::Int(_) | Type::Bool(_) | Type::Char(_) | Type::Float(_) => {
                    let type_ = arg.result_type();
                    dbg!(&arg);
                    let res_reg_raw = self.expression(arg);
                    dbg!(res_reg_raw);
                    let res_reg: IntRegister = res_reg_raw.expect("type is int").into();
                    if let Some(reg) = IntRegister::nth_param(int_cnt) {
                        param_regs.push(reg.to_reg());
                        self.use_reg(reg.to_reg(), Size::from(type_));
                        if res_reg != reg {
                            self.insert_movi(reg, res_reg, file!(), line!());
                        }
                    } else {
                        // no more params: spilling required
                        self.insert_with_comment(
                            Instruction::Store64(
                                res_reg,
                                IntRegisterPointer(IntRegister::R13, spill_cnt * 8),
                            ),
                            format!("{} byte param spill", Size::from(type_).byte_count()).into(),
                        );
                        spill_cnt += 1;
                        spill_param_size += 8;
                    }
                    int_cnt += 1;
                }
            }
        }

        self.curr_fn_mut().stack_allocs += spill_param_size;

        // perform function call
        let func_label = match node.func {
            "exit" => {
                // mark the current block as terminated (avoid useless jumps)
                self.curr_block_mut().is_terminated = true;
                "exit".into()
            }
            func => format!("main..{func}"),
        };
        self.insert_call(func_label);

        // restore the old list of used registers
        self.used_registers = used_regs_prev;

        let res_reg = match node.result_type {
            Type::Float(0) => Some(FloatRegister::F0.to_reg()),
            Type::Int(_) | Type::Char(_) | Type::Bool(_) | Type::Float(_) => {
                Some(IntRegister::R2.to_reg())
            }
            Type::Unit | Type::Never | Type::Unknown => None,
        };

        // restore all caller saved registers again
        self.restore_regs_after_call(res_reg, regs_on_stack)
    }
}

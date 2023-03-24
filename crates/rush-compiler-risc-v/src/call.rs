use std::{borrow::Cow, mem, rc::Rc, vec};

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

impl<'tree> Compiler<'tree> {
    /// Returns the instructions of a function prologue.
    /// Automatically sets up any stack allocations and saves `ra` and `fp`.
    /// Must be invoked after the fn body since the stack frame size must be known at this point.
    pub(crate) fn prologue(&mut self) -> Vec<(Instruction, Option<Cow<'static, str>>)> {
        // align frame size to 16 bytes
        Self::align(&mut self.curr_fn_mut().stack_allocs, 16);

        vec![
            (Instruction::Comment("begin prologue".into()), None),
            // allocate stack spacec
            (
                Instruction::Addi(
                    IntRegister::Sp,
                    IntRegister::Sp,
                    -self.curr_fn().stack_allocs - BASE_STACK_ALLOCATIONS,
                ),
                None,
            ),
            // save `fp` on the stack
            (
                Instruction::Sd(
                    IntRegister::Fp,
                    Pointer::Register(IntRegister::Sp, self.curr_fn().stack_allocs + 8),
                ),
                None,
            ),
            // save `ra` on the stack
            (
                Instruction::Sd(
                    IntRegister::Ra,
                    Pointer::Register(IntRegister::Sp, self.curr_fn().stack_allocs),
                ),
                None,
            ),
            // also offset the `fp` to start at the new frame
            (
                Instruction::Addi(
                    IntRegister::Fp,
                    IntRegister::Sp,
                    self.curr_fn().stack_allocs + BASE_STACK_ALLOCATIONS,
                ),
                None,
            ),
            (Instruction::Comment("end prologue".into()), None),
        ]
    }

    /// Inserts the instructions for a function epilogue.
    /// Places the instructions at the end of the epilogue label of the current function.
    pub(crate) fn epilogue(&mut self) {
        let epilogue_label = Rc::clone(&self.curr_fn().epilogue_label);
        self.insert_at(&epilogue_label);
        // restore `fp` from the stack
        self.insert(Instruction::Ld(
            IntRegister::Fp,
            Pointer::Register(IntRegister::Sp, self.curr_fn().stack_allocs + 8),
        ));
        // restore `ra` from the stack
        self.insert(Instruction::Ld(
            IntRegister::Ra,
            Pointer::Register(IntRegister::Sp, self.curr_fn().stack_allocs),
        ));
        // deallocate stack space
        self.insert(Instruction::Addi(
            IntRegister::Sp,
            IntRegister::Sp,
            self.curr_fn().stack_allocs + BASE_STACK_ALLOCATIONS,
        ));
        // return control back to caller
        self.insert(Instruction::Ret);
    }

    /// Compiles an [`AnalyzedCallExpr`].
    /// Prior to calling the target function, all currently used registers are saved on the stack.
    /// After the call has been performed, all previously saved registers are restored from memory.
    pub(crate) fn call_expr(&mut self, node: AnalyzedCallExpr<'tree>) -> Option<Register> {
        // before the function is called, all currently used registers are saved
        let regs_on_stack = self
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
        let mut spill_count = 0;

        for arg in node.args {
            match arg.result_type() {
                Type::Unit | Type::Never | Type::Unknown => {
                    self.expression(arg);
                }
                Type::Float(0) => {
                    let res_reg = self.expression(arg).expect("type is float");

                    if let Some(reg) = FloatRegister::nth_param(float_cnt) {
                        param_regs.push(reg.to_reg());
                        self.use_reg(reg.to_reg(), Size::Dword);
                    } else {
                        // no more param registers: spilling required
                        self.insert_with_comment(
                            Instruction::Fsd(
                                res_reg.into(),
                                Pointer::Register(IntRegister::Sp, spill_count * 8),
                            ),
                            format!("{} byte param spill", Size::Dword.byte_count(),).into(),
                        );
                        spill_count += 1;
                        spill_param_size += 8;
                    }
                    float_cnt += 1;
                }
                Type::Int(_) | Type::Bool(_) | Type::Char(_) | Type::Float(_) => {
                    let type_ = arg.result_type();
                    let res_reg = self.expression(arg).expect("`None` filtered above");
                    if let Some(reg) = IntRegister::nth_param(int_cnt) {
                        param_regs.push(reg.to_reg());
                        self.use_reg(reg.to_reg(), Size::from(type_));
                    } else {
                        // no more params: spilling required
                        self.insert_with_comment(
                            Instruction::Sd(
                                res_reg.into(),
                                Pointer::Register(IntRegister::Sp, spill_count * 8),
                            ),
                            format!("{} byte param spill", Size::from(type_).byte_count()).into(),
                        );
                        spill_count += 1;
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
                // mark the current block as terminated (avoid future useless jumps)
                self.curr_block_mut().is_terminated = true;
                "exit".into()
            }
            func => format!("main..{func}").into(),
        };
        self.insert(Instruction::Call(func_label));

        // restore the old list of used registers
        self.used_registers = used_regs_prev;

        let res_reg = match node.result_type {
            Type::Float(0) => Some(FloatRegister::Fa0.to_reg()),
            Type::Int(_) | Type::Char(_) | Type::Bool(_) | Type::Float(_) => {
                Some(IntRegister::A0.to_reg())
            }
            Type::Unit | Type::Never => None,
            Type::Unknown => unreachable!("analyzer would have failed"),
        };

        // restore all caller saved registers again
        self.restore_regs_after_call(res_reg, regs_on_stack)
    }
}

use rush_analyzer::InfixOp;

use crate::{
    condition::Condition,
    instruction::Instruction,
    register::{IntRegister, Register},
    value::{FloatValue, IntValue, Size, Value},
    Compiler,
};

impl<'src> Compiler<'src> {
    pub(crate) fn compile_infix(
        &mut self,
        lhs: Option<Register>,
        rhs: Option<Value>,
        op: InfixOp,
        is_char: bool,
    ) -> Option<Value> {
        match (lhs, rhs, op) {
            // `None` means a value of type `!` in this context, so don't do anything, as this
            // expression is unreachable at runtime
            (None, _, _) | (_, None, _) => None,
            // basic arithmetic for `int` and bitwise ops for `int` and `bool`
            (
                Some(Register::Int(left)),
                Some(Value::Int(right)),
                InfixOp::Plus
                | InfixOp::Minus
                | InfixOp::Mul
                | InfixOp::BitOr
                | InfixOp::BitAnd
                | InfixOp::BitXor,
            ) => {
                // if rhs is a register, it can be freed
                if matches!(&right, IntValue::Register(_)) {
                    self.used_registers.pop();
                }

                self.function_body.push(match op {
                    InfixOp::Plus => Instruction::Add(left.into(), right),
                    InfixOp::Minus => Instruction::Sub(left.into(), right),
                    InfixOp::Mul => Instruction::Imul(left.into(), right),
                    InfixOp::BitOr => Instruction::Or(left.into(), right),
                    InfixOp::BitAnd => Instruction::And(left.into(), right),
                    InfixOp::BitXor => Instruction::Xor(left.into(), right),
                    _ => unreachable!("this arm only matches with above ops"),
                });
                if is_char {
                    // truncate char to 7 bits
                    self.function_body
                        .push(Instruction::And(left.into(), 0x7f.into()));
                }
                Some(Value::Int(left.into()))
            }
            // integer division
            (Some(Register::Int(left)), Some(Value::Int(right)), InfixOp::Div | InfixOp::Rem) => {
                // make sure the rax and rdx registers are free
                let spilled_rax = self.spill_int_if_used(IntRegister::Rax);
                let spilled_rdx = self.spill_int_if_used(IntRegister::Rdx);

                let pop_rhs_reg = matches!(&right, IntValue::Register(_));

                // move lhs result into rax
                // analyzer guarantees `left` and `right` to be 8 bytes in size
                self.function_body
                    .push(Instruction::Mov(IntRegister::Rax.into(), left.into()));

                // get source operand
                let source = match right {
                    right @ IntValue::Ptr(_) => right,
                    right => {
                        // move rhs result into free register
                        let mut reg = self.get_tmp_register(Size::Qword);
                        // don't allow %rax and %rdx
                        if reg == IntRegister::Rax || reg == IntRegister::Rdx {
                            reg = reg.next()
                        }
                        self.function_body.push(Instruction::Mov(reg.into(), right));
                        reg.into()
                    }
                };

                // free the rhs register
                if pop_rhs_reg {
                    self.used_registers.pop();
                }

                // sign-extend lhs to 128 bits (required for IDIV)
                self.function_body.push(Instruction::Cqo);

                // divide
                self.function_body.push(Instruction::Idiv(source));

                // move result into result register
                self.function_body.push(Instruction::Mov(
                    left.into(),
                    // use either `%rax` or `%rdx` register, depending on operator
                    IntValue::Register(match op {
                        InfixOp::Div => IntRegister::Rax,
                        InfixOp::Rem => IntRegister::Rdx,
                        _ => unreachable!("this arm only matches with `/` or `%`"),
                    }),
                ));

                // reload spilled registers
                self.reload_int_if_used(spilled_rax);
                self.reload_int_if_used(spilled_rdx);

                Some(Value::Int(IntValue::Register(left)))
            }
            // integer shifts
            (Some(Register::Int(left)), Some(Value::Int(right)), InfixOp::Shl | InfixOp::Shr) => {
                // make sure the %rcx register is free
                let spilled_rcx = self.spill_int_if_used(IntRegister::Rcx);

                let rhs = match right {
                    IntValue::Immediate(num) => (num & 0xff).into(),
                    right => {
                        // if rhs is a register, free it
                        if let IntValue::Register(_) = right {
                            self.used_registers.pop();
                        }

                        // move rhs into %rcx
                        self.function_body
                            .push(Instruction::Mov(IntRegister::Rcx.into(), right));

                        // compare %rcx to 255
                        self.function_body
                            .push(Instruction::Cmp(IntRegister::Rcx.into(), 255.into()));

                        IntRegister::Cl.into()
                    }
                };

                // shift
                self.function_body.push(match op {
                    InfixOp::Shl => Instruction::Shl(left.into(), rhs),
                    InfixOp::Shr => Instruction::Sar(left.into(), rhs),
                    _ => unreachable!("this arm is only entered with `<<` or `>>` operator"),
                });

                // reload spilled register
                self.reload_int_if_used(spilled_rcx);

                Some(Value::Int(IntValue::Register(left)))
            }
            // int comparisons
            (
                Some(Register::Int(left)),
                Some(Value::Int(right)),
                InfixOp::Eq
                | InfixOp::Neq
                | InfixOp::Lt
                | InfixOp::Lte
                | InfixOp::Gt
                | InfixOp::Gte,
            ) => {
                // if rhs is a register, free it
                if let IntValue::Register(_) = right {
                    self.used_registers.pop();
                }

                // compare the sides
                self.function_body
                    .push(Instruction::Cmp(left.into(), right));

                // set the result
                let result_reg = left.in_byte_size();
                self.function_body.push(Instruction::SetCond(
                    Condition::try_from_op(op, true)
                        .expect("this block is only run with above ops"),
                    result_reg,
                ));

                Some(Value::Int(IntValue::Register(result_reg)))
            }
            // float arithmetic
            (
                Some(Register::Float(left)),
                Some(Value::Float(right)),
                InfixOp::Plus | InfixOp::Minus | InfixOp::Mul | InfixOp::Div,
            ) => {
                // if rhs is a register, it can be freed
                if matches!(&right, FloatValue::Register(_)) {
                    self.used_float_registers.pop();
                }

                self.function_body.push(match op {
                    InfixOp::Plus => Instruction::Addsd(left.into(), right),
                    InfixOp::Minus => Instruction::Subsd(left.into(), right),
                    InfixOp::Mul => Instruction::Mulsd(left.into(), right),
                    InfixOp::Div => Instruction::Divsd(left.into(), right),
                    _ => unreachable!("this arm only matches with above ops"),
                });

                Some(Value::Float(left.into()))
            }
            // float comparisons
            (
                Some(Register::Float(left)),
                Some(Value::Float(right)),
                InfixOp::Eq
                | InfixOp::Neq
                | InfixOp::Lt
                | InfixOp::Lte
                | InfixOp::Gt
                | InfixOp::Gte,
            ) => {
                // if rhs is a register, it can be freed
                if matches!(&right, FloatValue::Register(_)) {
                    self.used_float_registers.pop();
                }

                // free the left register (result needs an IntRegister)
                self.used_float_registers.pop();

                // compare the sides
                self.function_body.push(Instruction::Ucomisd(left, right));

                // set the result
                let reg = self.get_free_register(Size::Byte);
                self.function_body.push(Instruction::SetCond(
                    Condition::try_from_op(op, false)
                        .expect("this block is only run with above ops"),
                    reg,
                ));

                // consider unordered results for equality checks
                if matches!(op, InfixOp::Eq | InfixOp::Lt | InfixOp::Lte) {
                    // save parity
                    let parity_reg = self.get_tmp_register(Size::Byte);
                    self.function_body
                        .push(Instruction::SetCond(Condition::NotParity, parity_reg));

                    // both results must be true (result must not be unordered)
                    self.function_body
                        .push(Instruction::And(reg.into(), parity_reg.into()));
                } else if op == InfixOp::Neq {
                    // save parity
                    let parity_reg = self.get_tmp_register(Size::Byte);
                    self.function_body
                        .push(Instruction::SetCond(Condition::Parity, parity_reg));

                    // either result can be true (result can be unordered)
                    self.function_body
                        .push(Instruction::Or(reg.into(), parity_reg.into()));
                }

                Some(Value::Int(IntValue::Register(reg)))
            }
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

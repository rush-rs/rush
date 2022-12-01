use rush_analyzer::InfixOp;

use crate::{
    instruction::{Condition, Instruction},
    register::IntRegister,
    value::{FloatValue, IntValue, Size, Value},
    Compiler,
};

impl<'src> Compiler<'src> {
    pub(crate) fn compile_infix(
        &mut self,
        lhs: Option<Value>,
        rhs: Option<Value>,
        op: InfixOp,
    ) -> Option<Value> {
        match (lhs, rhs, op) {
            // `None` means a value of type `!` in this context, so don't do anything, as this
            // expression is unreachable at runtime
            (None, _, _) | (_, None, _) => None,
            // basic arithmetic for `int` and bitwise ops for `int` and `bool`
            (
                Some(Value::Int(left)),
                Some(Value::Int(right)),
                InfixOp::Plus
                | InfixOp::Minus
                | InfixOp::Mul
                | InfixOp::BitOr
                | InfixOp::BitAnd
                | InfixOp::BitXor,
            ) => {
                // if two registers were in use before, one can be freed afterwards
                let pop_rhs_reg = matches!(
                    (&left, &right),
                    (IntValue::Register(_), IntValue::Register(_))
                );

                let (left, right) = match (left, right) {
                    // when one side is already a register, use that in matching size
                    (IntValue::Register(left), right)
                    // note the swapped sides here
                    | (right, IntValue::Register(left)) => {
                        (
                            match &right {
                                IntValue::Register(reg) => left.in_size(reg.size()),
                                IntValue::Ptr(ptr) => left.in_size(ptr.size),
                                IntValue::Immediate(_) => left,
                            },
                            right,
                        )
                    }
                    // else move the left value into a free register and use that
                    (left, right) => {
                        let reg = self.get_free_register(match &right {
                            IntValue::Ptr(ptr) => ptr.size,
                            _ => Size::Qword,
                        });
                        self.function_body.push(Instruction::Mov(reg.into(), left));
                        (reg, right)
                    }
                };

                self.function_body.push(match op {
                    InfixOp::Plus => Instruction::Add(left.into(), right),
                    InfixOp::Minus => Instruction::Sub(left.into(), right),
                    InfixOp::Mul => Instruction::Imul(left.into(), right),
                    InfixOp::BitOr => Instruction::Or(left.into(), right),
                    InfixOp::BitAnd => Instruction::And(left.into(), right),
                    InfixOp::BitXor => Instruction::Xor(left.into(), right),
                    _ => unreachable!("this arm only matches with above ops"),
                });
                if pop_rhs_reg {
                    // free the rhs register
                    self.used_registers.pop();
                }
                Some(Value::Int(left.into()))
            }
            // integer division
            (Some(Value::Int(left)), Some(Value::Int(right)), InfixOp::Div | InfixOp::Rem) => {
                // make sure the rax and rdx registers are free
                let spilled_rax = self.spill_int_if_used(IntRegister::Rax);
                let spilled_rdx = self.spill_int_if_used(IntRegister::Rdx);

                let mut pop_rhs_reg = false;
                let result_reg = match (&left, &right) {
                    (IntValue::Register(reg), IntValue::Register(_)) => {
                        pop_rhs_reg = true;
                        *reg
                    }
                    (IntValue::Register(reg), _) | (_, IntValue::Register(reg)) => *reg,
                    _ => self.get_free_register(Size::Qword),
                };

                // move lhs result into rax
                // analyzer guarantees `left` and `right` to be 8 bytes in size
                self.function_body
                    .push(Instruction::Mov(IntRegister::Rax.into(), left));

                // get source operand
                let source = match right {
                    right @ IntValue::Ptr(_) => right,
                    right => {
                        // move rhs result into free register
                        let mut reg = self
                            .used_registers
                            .last()
                            .map_or(IntRegister::Rdi, |reg| reg.next());
                        // don't allow %rdx
                        if reg == IntRegister::Rdx {
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
                    result_reg.into(),
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

                Some(Value::Int(IntValue::Register(result_reg)))
            }
            // integer shifts
            (Some(Value::Int(left)), Some(Value::Int(right)), InfixOp::Shl | InfixOp::Shr) => {
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

                let lhs = match left {
                    // when left side already uses a register, use that register
                    IntValue::Register(reg) => reg,
                    // else move into free one
                    left => {
                        let reg = self.get_free_register(Size::Qword);
                        self.function_body.push(Instruction::Mov(reg.into(), left));
                        reg
                    }
                };

                // shift
                self.function_body.push(match op {
                    InfixOp::Shl => Instruction::Shl(lhs.into(), rhs),
                    InfixOp::Shr => Instruction::Sar(lhs.into(), rhs),
                    _ => unreachable!("this arm is only entered with `<<` or `>>` operator"),
                });

                // reload spilled register
                self.reload_int_if_used(spilled_rcx);

                Some(Value::Int(IntValue::Register(lhs)))
            }
            // int comparisons
            (
                Some(Value::Int(left)),
                Some(Value::Int(right)),
                InfixOp::Eq
                | InfixOp::Neq
                | InfixOp::Lt
                | InfixOp::Lte
                | InfixOp::Gt
                | InfixOp::Gte,
            ) => {
                let (lhs, rhs, reg) = match (left, right) {
                    (IntValue::Register(reg), right @ IntValue::Register(_)) => {
                        // free right register
                        self.used_registers.pop();

                        // use left register for result
                        (reg.into(), right, reg.in_byte_size())
                    }
                    (IntValue::Register(reg), right) => (reg.into(), right, reg.in_byte_size()),
                    (left, IntValue::Register(reg)) => (left, reg.into(), reg.in_byte_size()),
                    (left, IntValue::Ptr(ptr)) => {
                        let reg = self.get_free_register(ptr.size);
                        self.function_body.push(Instruction::Mov(reg.into(), left));
                        (reg.into(), ptr.into(), reg.in_byte_size())
                    }
                    (IntValue::Ptr(ptr), right @ IntValue::Immediate(_)) => {
                        let reg = self.get_free_register(ptr.size);
                        (ptr.into(), right, reg.in_byte_size())
                    }
                    (IntValue::Immediate(left), IntValue::Immediate(right)) => {
                        return Some(Value::Int(IntValue::Immediate(match op {
                            InfixOp::Eq => left == right,
                            InfixOp::Neq => left != right,
                            InfixOp::Lt => left < right,
                            InfixOp::Gt => left > right,
                            InfixOp::Lte => left <= right,
                            InfixOp::Gte => left >= right,
                            _ => unreachable!("this block is only run with above ops"),
                        }
                            as i64)))
                    }
                };

                // compare the sides
                self.function_body.push(Instruction::Cmp(lhs, rhs));

                // set the result
                self.function_body.push(Instruction::SetCond(
                    match op {
                        InfixOp::Eq => Condition::Equal,
                        InfixOp::Neq => Condition::NotEqual,
                        InfixOp::Lt => Condition::Less,
                        InfixOp::Gt => Condition::Greater,
                        InfixOp::Lte => Condition::LessOrEqual,
                        InfixOp::Gte => Condition::GreaterOrEqual,
                        _ => unreachable!("this block is only run with above ops"),
                    },
                    reg,
                ));

                Some(Value::Int(IntValue::Register(reg)))
            }
            // float arithmetic
            (
                Some(Value::Float(left)),
                Some(Value::Float(right)),
                InfixOp::Plus | InfixOp::Minus | InfixOp::Mul | InfixOp::Div,
            ) => {
                // if two registers were in use before, one can be freed afterwards
                let pop_rhs_reg = matches!(
                    (&left, &right),
                    (FloatValue::Register(_), FloatValue::Register(_))
                );

                let (left, right) = match (left, right) {
                    // when one side is already a register, use that
                    (FloatValue::Register(left), right)
                    // note the swapped sides here
                    | (right, FloatValue::Register(left)) => {
                        (left, right)
                    }
                    // else move the left value into a free register and use that
                    (left, right) => {
                        let reg = self.get_free_float_register();
                        self.function_body.push(Instruction::Movsd(reg.into(), left));
                        (reg, right)
                    }
                };

                self.function_body.push(match op {
                    InfixOp::Plus => Instruction::Addsd(left.into(), right),
                    InfixOp::Minus => Instruction::Subsd(left.into(), right),
                    InfixOp::Mul => Instruction::Mulsd(left.into(), right),
                    InfixOp::Div => Instruction::Divsd(left.into(), right),
                    _ => unreachable!("this arm only matches with above ops"),
                });
                if pop_rhs_reg {
                    // free the rhs register
                    self.used_registers.pop();
                }
                Some(Value::Float(left.into()))
            }
            // float comparisons
            (
                Some(Value::Float(left)),
                Some(Value::Float(right)),
                InfixOp::Eq
                | InfixOp::Neq
                | InfixOp::Lt
                | InfixOp::Lte
                | InfixOp::Gt
                | InfixOp::Gte,
            ) => {
                let (lhs, rhs) = match (left, right) {
                    (FloatValue::Register(left), right @ FloatValue::Register(_)) => {
                        // free both registers
                        self.used_float_registers.pop();
                        self.used_float_registers.pop();

                        (left, right)
                    }
                    (FloatValue::Register(left), right) => {
                        // free the left register
                        self.used_float_registers.pop();

                        (left, right)
                    }
                    (left, right @ FloatValue::Register(_)) => {
                        // move left side into free register
                        let reg = self.get_tmp_float_register();
                        self.function_body
                            .push(Instruction::Movsd(reg.into(), left));

                        // free the right register
                        self.used_float_registers.pop();

                        (reg, right)
                    }
                    (left, right) => {
                        // move left side into free register
                        let reg = self.get_tmp_float_register();
                        self.function_body
                            .push(Instruction::Movsd(reg.into(), left));

                        (reg, right)
                    }
                };

                // compare the sides
                self.function_body.push(Instruction::Ucomisd(lhs, rhs));

                // set the result
                let reg = self.get_free_register(Size::Byte);
                self.function_body.push(Instruction::SetCond(
                    match op {
                        InfixOp::Eq => Condition::Equal,
                        InfixOp::Neq => Condition::NotEqual,
                        InfixOp::Lt => Condition::Below,
                        InfixOp::Gt => Condition::Above,
                        InfixOp::Lte => Condition::BelowOrEqual,
                        InfixOp::Gte => Condition::AboveOrEqual,
                        _ => unreachable!("this block is only run with above ops"),
                    },
                    reg,
                ));

                // consider unordered results for equality checks
                if op == InfixOp::Eq {
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

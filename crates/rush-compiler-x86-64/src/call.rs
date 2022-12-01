use std::mem;

use rush_analyzer::{ast::AnalyzedExpression, Type};

use crate::{
    instruction::Instruction,
    register::{FloatRegister, IntRegister, FLOAT_PARAM_REGISTERS, INT_PARAM_REGISTERS},
    value::{FloatValue, IntValue, Pointer, Size, Value},
    Compiler,
};

impl<'src> Compiler<'src> {
    pub(crate) fn call_func(
        &mut self,
        result_type: Type,
        func: String,
        args: Vec<AnalyzedExpression<'src>>,
    ) -> Option<Value> {
        let prev_used_registers = mem::take(&mut self.used_registers);
        let prev_used_float_registers = mem::take(&mut self.used_float_registers);

        // save currently used caller-saved registers on stack
        let mut saved_register_pointers = vec![];
        let mut saved_float_register_pointers = vec![];
        for reg in prev_used_registers
            .iter()
            .filter(|reg| reg.is_caller_saved())
        {
            saved_register_pointers.push(self.spill_int(*reg));
        }
        for reg in &prev_used_float_registers {
            saved_float_register_pointers.push(self.push_to_stack(
                Size::Qword,
                Value::Float((*reg).into()),
                Some(format!("8 byte spill: {reg}")),
            ));
        }

        self.in_args += 1;

        // compile arg exprs
        let mut int_register_index = 0;
        let mut float_register_index = 0;
        let mut memory_offset = 0;
        for arg in args {
            match self.expression(arg) {
                None => {}
                Some(Value::Int(value)) => {
                    match value {
                        IntValue::Register(reg)
                            if int_register_index < INT_PARAM_REGISTERS.len() =>
                        {
                            debug_assert_eq!(
                                reg.in_qword_size(),
                                INT_PARAM_REGISTERS[int_register_index]
                            );
                        }
                        src @ (IntValue::Ptr(_) | IntValue::Immediate(_))
                            if int_register_index < INT_PARAM_REGISTERS.len() =>
                        {
                            let reg = self.get_free_register(match &src {
                                IntValue::Ptr(ptr) => ptr.size,
                                _ => Size::Qword,
                            });
                            debug_assert_eq!(
                                reg.in_qword_size(),
                                INT_PARAM_REGISTERS[int_register_index]
                            );
                            self.function_body.push(Instruction::Mov(reg.into(), src));
                        }
                        src @ (IntValue::Register(_) | IntValue::Immediate(_)) => {
                            self.function_body.push(Instruction::Mov(
                                Pointer::new(Size::Qword, IntRegister::Rsp, memory_offset.into())
                                    .into(),
                                src,
                            ));
                            memory_offset += 8;
                        }
                        IntValue::Ptr(ptr) => {
                            let reg = self.get_tmp_register(ptr.size);
                            self.function_body
                                .push(Instruction::Mov(reg.into(), ptr.into()));
                            self.function_body.push(Instruction::Mov(
                                Pointer::new(Size::Qword, IntRegister::Rsp, memory_offset.into())
                                    .into(),
                                reg.into(),
                            ));
                            memory_offset += 8;
                        }
                    }
                    int_register_index += 1;
                }
                Some(Value::Float(value)) => {
                    match value {
                        FloatValue::Register(reg) => {
                            if float_register_index < FLOAT_PARAM_REGISTERS.len() {
                                debug_assert_eq!(reg, FLOAT_PARAM_REGISTERS[float_register_index]);
                            } else {
                                self.function_body.push(Instruction::Movsd(
                                    Pointer::new(
                                        Size::Qword,
                                        IntRegister::Rsp,
                                        memory_offset.into(),
                                    )
                                    .into(),
                                    reg.into(),
                                ));
                                memory_offset += 8;
                            }
                        }
                        FloatValue::Ptr(ptr) => {
                            if float_register_index < FLOAT_PARAM_REGISTERS.len() {
                                let reg = self.get_free_float_register();
                                debug_assert_eq!(reg, FLOAT_PARAM_REGISTERS[float_register_index]);
                                self.function_body
                                    .push(Instruction::Movsd(reg.into(), ptr.into()));
                            } else {
                                let reg = self.get_tmp_float_register();
                                self.function_body
                                    .push(Instruction::Movsd(reg.into(), ptr.into()));
                                self.function_body.push(Instruction::Movsd(
                                    Pointer::new(
                                        Size::Qword,
                                        IntRegister::Rsp,
                                        memory_offset.into(),
                                    )
                                    .into(),
                                    reg.into(),
                                ));
                                memory_offset += 8;
                            }
                        }
                    }
                    float_register_index += 1;
                }
            }
        }

        // allocate the required param memory, but do not modify `self.stack_pointer`
        self.frame_size += memory_offset;

        // call function
        self.function_body.push(Instruction::Call(func));

        // move result to free register
        self.in_args -= 1;
        self.used_registers = prev_used_registers.clone();
        self.used_float_registers = prev_used_float_registers.clone();
        let result_reg = match result_type {
            Type::Unit | Type::Never => None,
            Type::Int | Type::Char | Type::Bool => {
                let size = Size::try_from(result_type).expect("int, char and bool have a size");
                let reg = self.get_free_register(size);
                self.function_body.push(Instruction::Mov(
                    reg.into(),
                    IntRegister::Rax.in_size(size).into(),
                ));
                Some(Value::Int(reg.into()))
            }
            Type::Float => {
                let reg = self.get_free_float_register();
                self.function_body
                    .push(Instruction::Movsd(reg.into(), FloatRegister::Xmm0.into()));
                Some(Value::Float(reg.into()))
            }
            Type::Unknown => unreachable!("the analyzer guarantees one of the above to match"),
        };

        // restore previously used caller-saved registers from stack
        for (reg, ptr) in prev_used_float_registers
            .into_iter()
            .zip(saved_float_register_pointers)
            .rev()
        {
            self.pop_from_stack(
                ptr,
                Value::Float(reg.into()),
                Some(format!("8 byte reload: {reg}")),
            );
        }
        for (reg, ptr) in prev_used_registers
            .into_iter()
            .filter(|reg| reg.is_caller_saved())
            .collect::<Vec<_>>() // required to know size
            .into_iter()
            .zip(saved_register_pointers)
            .rev()
        {
            self.reload_int(ptr, reg);
        }

        result_reg
    }
}

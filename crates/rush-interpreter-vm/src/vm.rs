use std::fmt::Display;

use crate::{instruction::Instruction, value::Value};

pub(crate) struct Vm {
    /// Stores the program's globals.
    globals: Vec<Value>,
    /// Working memory for temporary values
    stack: Vec<Value>,
    /// Holds information about the current position (like ip / fp).
    call_stack: Vec<CallFrame>,
}

#[derive(Debug, Default)]
struct CallFrame {
    /// Variable bindings for the current function.
    mem: Vec<Value>,
    /// Specifies the instruction pointer relative to the function
    ip: usize,
    /// Specifies the function pointer
    fp: usize,
}

const STACK_LIMIT: usize = 1000;
const CALL_STACK_LIMIT: usize = 1000;

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug)]
pub struct RuntimeError {
    pub kind: RuntimeErrorKind,
    pub msg: String,
}

impl RuntimeError {
    pub fn new(kind: RuntimeErrorKind, msg: String) -> Self {
        Self { kind, msg }
    }
}

#[derive(Debug)]
pub enum RuntimeErrorKind {
    StackOverflow,
    Arithmetic,
}

impl Display for RuntimeErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::StackOverflow => "StackOverflowError",
                Self::Arithmetic => "ArithmeticError",
            }
        )
    }
}

impl Vm {
    pub(crate) fn new() -> Self {
        Self {
            globals: vec![],
            stack: vec![],
            // start execution at the first instruction of the prelude
            call_stack: vec![CallFrame::default()],
        }
    }

    /// Adds a value to the stack.
    /// If this operation would exceed the stack limit, an error is returned.
    fn push(&mut self, val: Value) -> Result<()> {
        match self.stack.len() > STACK_LIMIT {
            true => Err(RuntimeError::new(
                RuntimeErrorKind::StackOverflow,
                format!("maximum stack size of {STACK_LIMIT} exceeded"),
            )),
            false => {
                self.stack.push(val);
                Ok(())
            }
        }
    }

    #[inline]
    /// Removes the top stack element and returns its value.
    fn pop(&mut self) -> Value {
        self.stack.pop().expect("pop is always called safely")
    }

    #[inline]
    fn call_frame(&self) -> &CallFrame {
        self.call_stack.last().expect("there is always the root")
    }

    #[inline]
    fn call_frame_mut(&mut self) -> &mut CallFrame {
        self.call_stack
            .last_mut()
            .expect("there is always the root")
    }

    pub(crate) fn run(&mut self, program: Vec<Vec<Instruction>>) -> Result<i64> {
        while self.call_frame().ip < program[self.call_frame().fp].len() {
            let instruction = &program[self.call_frame().fp][self.call_frame().ip];

            // TODO: remove this debug comment eventually
            /* println!(
                "[{:02}/{:02}] {:10} | {}",
                self.call_frame().fp,
                self.call_frame().ip,
                instruction.to_string(),
                self.stack
                    .iter()
                    .rev()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(",")
            ); */

            // if the current instruction exists the VM, terminate execution
            if let Some(code) = self.instruction(instruction)? {
                return Ok(code);
            };
        }

        #[cfg(debug_assertions)]
        assert_eq!(self.stack.last(), Some(&Value::Unit));

        Ok(0)
    }

    fn instruction(&mut self, inst: &Instruction) -> Result<Option<i64>> {
        match inst {
            // indexing is safe here because only valid instructions are ran
            Instruction::Push(value) => self.push(*value)?,
            Instruction::Pop => {
                self.pop();
            }
            Instruction::Jmp(idx) => {
                self.call_frame_mut().ip = *idx;
                return Ok(None);
            }
            Instruction::JmpCond(idx) => {
                // if the value on the stack is `false`, perform the jump
                if !self.pop().into_bool() {
                    self.call_frame_mut().ip = *idx;
                    return Ok(None);
                }
            }
            Instruction::SetVar(idx) => {
                let val = self.pop();
                // if there is already an entry in the memory, use it
                // otherwise, new memory is allocated
                match self.call_frame().mem.len() < *idx + 1 {
                    true => self.call_frame_mut().mem.push(val),
                    false => self.call_frame_mut().mem[*idx] = val,
                }
            }
            Instruction::GetVar(idx) => self.push(self.call_frame().mem[*idx])?,
            Instruction::SetGlob(idx) => {
                let val = self.pop();
                match self.globals.len() < *idx + 1 {
                    true => self.globals.push(val),
                    false => self.globals[*idx] = val,
                }
            }
            Instruction::GetGlob(idx) => self.push(self.globals[*idx])?,
            Instruction::Call(idx) => {
                if self.call_stack.len() > CALL_STACK_LIMIT {
                    return Err(RuntimeError::new(
                        RuntimeErrorKind::StackOverflow,
                        format!("maximum call-stack size of {CALL_STACK_LIMIT} was exceeded"),
                    ));
                }

                // sets the function pointer and instruction pointer to the callee
                self.call_stack.push(CallFrame {
                    ip: 0,
                    fp: *idx,
                    mem: vec![],
                });

                // must return because the ip would be incremented otherwise
                return Ok(None);
            }
            Instruction::Cast(to) => {
                let from = self.pop();
                self.push(from.cast(*to))?
            }
            Instruction::Ret => {
                self.call_stack.pop();
            }
            Instruction::Exit => {
                return Ok(Some(self.pop().into_int()));
            }
            Instruction::Neg => {
                let top = self.pop();
                self.push(top.neg())?;
            }
            Instruction::Not => {
                let top = self.pop();
                self.push(top.not())?;
            }
            Instruction::Add => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.add(rhs)?)?;
            }
            Instruction::Sub => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.sub(rhs)?)?;
            }
            Instruction::Mul => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.mul(rhs)?)?;
            }
            Instruction::Pow => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.pow(rhs)?)?;
            }
            Instruction::Div => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.div(rhs)?)?;
            }
            Instruction::Rem => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.rem(rhs)?)?;
            }
            Instruction::Eq => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.eq(rhs))?;
            }
            Instruction::Ne => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.ne(rhs))?;
            }
            Instruction::Lt => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.lt(rhs))?;
            }
            Instruction::Le => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.le(rhs))?;
            }
            Instruction::Gt => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.gt(rhs))?;
            }
            Instruction::Ge => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.ge(rhs))?;
            }
            Instruction::Shl => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.shl(rhs)?)?;
            }
            Instruction::Shr => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.shr(rhs)?)?;
            }
            Instruction::BitOr => todo!(),
            Instruction::BitAnd => todo!(),
            Instruction::BitXor => todo!(),
            Instruction::And => todo!(),
            Instruction::Or => todo!(),
        }
        self.call_frame_mut().ip += 1;
        Ok(None)
    }
}

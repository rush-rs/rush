use std::{
    fmt::{self, Display, Formatter},
    thread::sleep,
    time::Duration,
};

use crate::{
    instruction::{Instruction, Program},
    value::{Pointer, Value},
};

//const MEM_SIZE: usize = 1024;
const STACK_LIMIT: usize = 1024;
const CALL_STACK_LIMIT: usize = 1024;

pub struct Vm<const MEM_SIZE: usize> {
    /// Working memory for temporary values
    stack: Vec<Value>,
    /// Linear memory for variables.
    mem: [Option<Value>; MEM_SIZE],
    /// The memory pointer points to the last free location in memory.
    /// The value is always positive, but using `isize` does not require casts.
    mem_ptr: isize,
    /// Holds information about the current position (like ip / fp).
    call_stack: Vec<CallFrame>,
}

#[derive(Debug, Default)]
struct CallFrame {
    /// Specifies the instruction pointer relative to the function
    ip: usize,
    /// Specifies the function pointer
    fp: usize,
}

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug)]
pub struct RuntimeError {
    pub kind: RuntimeErrorKind,
    pub msg: String,
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.msg)
    }
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
    OutOfMem,
}

impl Display for RuntimeErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::StackOverflow => "StackOverflowError",
                Self::Arithmetic => "ArithmeticError",
                Self::OutOfMem => "OutOfMemory",
            }
        )
    }
}

impl<const MEM_SIZE: usize> Default for Vm<MEM_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MEM_SIZE: usize> Vm<MEM_SIZE> {
    pub fn new() -> Self {
        Self {
            stack: vec![],
            mem: [None; MEM_SIZE],
            mem_ptr: 0,
            // start execution at the first instruction of the prelude
            call_stack: vec![CallFrame::default()],
        }
    }

    /// Adds a value to the stack.
    /// If this operation would exceed the stack limit, an error is returned.
    fn push(&mut self, val: Value) -> Result<()> {
        match self.stack.len() >= STACK_LIMIT {
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
        self.call_stack
            .last()
            .expect("there is always at least one call frame")
    }

    #[inline]
    fn call_frame_mut(&mut self) -> &mut CallFrame {
        self.call_stack
            .last_mut()
            .expect("there is always at least one call frame")
    }

    /// Runs the specified program but includes debug prints after each instruction.
    /// Also accepts the `clock_hz` parameter which specifies the speed of the VM.
    pub fn debug_run(&mut self, program: Program, clock_hz: u64) -> Result<i64> {
        while self.call_frame().ip < program.0[self.call_frame().fp].len() {
            let instruction = &program.0[self.call_frame().fp][self.call_frame().ip];

            println!(
                "[{fp:02}/{ip:02}] {instr:15} | {stack:>40} | mp = {mp} | clstck = {call_stack} | {mem:?}",
                fp = self.call_frame().fp,
                ip = self.call_frame().ip,
                instr = instruction.to_string(),
                stack = self
                    .stack
                    .iter()
                    .rev()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
                mp = self.mem_ptr,
                call_stack = self.call_stack.len(),
                mem = &self.mem[0..=self.mem_ptr as usize],
                    /* .iter()
                    .map(|v| match v {
                        Some(v) => v.to_string(),
                        None => "NONE".to_string(),
                    })
                    .collect::<Vec<String>>()
                    .join(", ") */
            );
            sleep(Duration::from_millis(1000 / clock_hz));

            // if the current instruction exists the VM, terminate execution
            if let Some(code) = self.run_instruction(instruction)? {
                return Ok(code);
            };
        }
        Ok(0)
    }

    pub fn run(&mut self, program: Program) -> Result<i64> {
        while self.call_frame().ip < program.0[self.call_frame().fp].len() {
            let instruction = &program.0[self.call_frame().fp][self.call_frame().ip];

            // if the current instruction exists the VM, terminate execution
            if let Some(code) = self.run_instruction(instruction)? {
                return Ok(code);
            };
        }

        Ok(0)
    }

    fn run_instruction(&mut self, inst: &Instruction) -> Result<Option<i64>> {
        match inst {
            Instruction::Nop => {}
            Instruction::Push(value) => self.push(*value)?,
            Instruction::Drop => {
                self.pop();
            }
            Instruction::Clone => self
                .stack
                .push(*self.stack.last().expect("clone is always called safely")),
            Instruction::Jmp(idx) => {
                self.call_frame_mut().ip = *idx;
                return Ok(None);
            }
            Instruction::JmpFalse(idx) => {
                // if the value on the stack is `false`, perform the jump
                if !self.pop().unwrap_bool() {
                    self.call_frame_mut().ip = *idx;
                    return Ok(None);
                }
            }
            Instruction::SetVarImm(_) | Instruction::SetVar => {
                let val = self.pop();

                let addr = match inst {
                    Instruction::SetVarImm(Pointer::Rel(offset)) => {
                        (self.mem_ptr + offset - 1) as usize
                    }
                    Instruction::SetVarImm(Pointer::Abs(addr)) => *addr,
                    _ => match self.pop().unwrap_ptr() {
                        Pointer::Rel(offset) => (self.mem_ptr + offset - 1) as usize,
                        Pointer::Abs(addr) => addr,
                    },
                };

                // if there is already an entry in the memory, use it.
                // otherwise, new memory is allocated
                match self.mem.get_mut(addr) {
                    Some(var) => *var = Some(val),
                    None => self.mem[addr] = Some(val),
                }
            }
            Instruction::GetVar => {
                let addr = match self.pop().unwrap_ptr() {
                    Pointer::Rel(offset) => (self.mem_ptr + offset - 1) as usize,
                    Pointer::Abs(addr) => addr,
                };
                self.push(self.mem[addr].expect("variables are always initialized"))?
            }
            Instruction::Call(idx) => {
                if self.call_stack.len() >= CALL_STACK_LIMIT {
                    return Err(RuntimeError::new(
                        RuntimeErrorKind::StackOverflow,
                        format!("maximum call-stack size of {CALL_STACK_LIMIT} was exceeded"),
                    ));
                }

                // sets the function pointer and instruction pointer to the callee
                self.call_stack.push(CallFrame { ip: 0, fp: *idx });

                // must return because the ip would be incremented otherwise
                return Ok(None);
            }
            Instruction::SetMp(offset) => {
                self.mem_ptr += offset;
                if self.mem_ptr < 0 || self.mem_ptr as usize >= MEM_SIZE {
                    return Err(RuntimeError::new(RuntimeErrorKind::OutOfMem, format!("Out of memory: the memory limit of {MEM_SIZE} cells was exceeded")))
                }
            }
            Instruction::RelToAddr(offset) => {
                let addr = (self.mem_ptr + offset - 1) as usize;
                self.push(Value::Ptr(Pointer::Abs(addr)))?;
            }
            Instruction::Ret => {
                self.call_stack.pop();
            }
            Instruction::Cast(to) => {
                let from = self.pop();
                self.push(from.cast(*to))?
            }
            Instruction::Exit => {
                return Ok(Some(self.pop().unwrap_int()));
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
                self.push(lhs.add(rhs))?;
            }
            Instruction::Sub => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.sub(rhs))?;
            }
            Instruction::Mul => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.mul(rhs))?;
            }
            Instruction::Pow => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.pow(rhs))?;
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
            Instruction::BitOr => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.bit_or(rhs))?;
            }
            Instruction::BitAnd => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.bit_and(rhs))?;
            }
            Instruction::BitXor => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.push(lhs.bit_xor(rhs))?;
            }
        }
        self.call_frame_mut().ip += 1;
        Ok(None)
    }
}

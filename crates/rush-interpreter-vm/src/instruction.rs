use std::fmt::{self, Display, Formatter};

use rush_analyzer::{AssignOp, InfixOp, PrefixOp, Type as AnalyzerType};

use crate::Value;

pub struct Program(pub(crate) Vec<Vec<Instruction>>);

impl Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let out = self
            .0
            .iter()
            .enumerate()
            .map(|(idx, func)| {
                let label = match idx {
                    0 => "prelude".to_string(),
                    1 => "main".to_string(),
                    idx => idx.to_string(),
                };
                format!(
                    "{label}: {}\n",
                    func.iter()
                        .enumerate()
                        .map(|(idx, i)| format!("\n [{idx:02}]    {i}"))
                        .collect::<String>()
                )
            })
            .collect::<String>();
        write!(f, "{out}")
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Type {
    Int,
    Bool,
    Char,
    Float,
}

impl From<AnalyzerType> for Type {
    fn from(src: AnalyzerType) -> Self {
        match src {
            AnalyzerType::Int => Self::Int,
            AnalyzerType::Float => Self::Float,
            AnalyzerType::Bool => Self::Bool,
            AnalyzerType::Char => Self::Char,
            _ => unreachable!("cannot convert from these types"),
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Type::Int => "int",
                Type::Bool => "bool",
                Type::Char => "char",
                Type::Float => "float",
            }
        )
    }
}

#[derive(Debug)]
pub enum Instruction {
    /// Adds a new constant to stack.
    Push(Value),
    /// Pops the top-most value off the stack and discards the value.
    Drop,
    /// Calls a function (specified by index).
    Call(usize),
    /// Returns from the current function call.
    Ret,
    /// Special instruction for exit calls.
    Exit,
    /// Jumps to the specified index.
    Jmp(usize),
    /// Jumps to the specified index if the value on the stack is `false`.
    JmpFalse(usize),
    /// Pops the top element off the stack and binds it to the specified number.
    SetVar(usize),
    /// Retrieves the variable with the specified index and places it on top of the stack.
    GetVar(usize),
    // Pops the top element off the stack and sets it as a global using the specified index.
    SetGlobal(usize),
    /// Retrieves the global with the specified index and places it on top of the stack.
    GetGlobal(usize),
    /// Cast the current item on the stack to the specified type.
    Cast(Type),

    // Prefix operators
    Neg,
    Not,

    // Infix operators
    Add,
    Sub,
    Mul,
    Pow,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Shl,
    Shr,
    BitOr,
    BitAnd,
    BitXor,
}

impl From<InfixOp> for Instruction {
    fn from(src: InfixOp) -> Self {
        match src {
            InfixOp::Plus => Self::Add,
            InfixOp::Minus => Self::Sub,
            InfixOp::Mul => Self::Mul,
            InfixOp::Div => Self::Div,
            InfixOp::Rem => Self::Rem,
            InfixOp::Pow => Self::Pow,
            InfixOp::Eq => Self::Eq,
            InfixOp::Neq => Self::Ne,
            InfixOp::Lt => Self::Lt,
            InfixOp::Gt => Self::Gt,
            InfixOp::Lte => Self::Le,
            InfixOp::Gte => Self::Ge,
            InfixOp::Shl => Self::Shl,
            InfixOp::Shr => Self::Shr,
            InfixOp::BitOr => Self::BitOr,
            InfixOp::BitAnd => Self::BitAnd,
            InfixOp::BitXor => Self::BitXor,
            _ => unreachable!("logical operators are handled separately"),
        }
    }
}

impl TryFrom<AssignOp> for Instruction {
    type Error = ();

    fn try_from(src: AssignOp) -> Result<Self, Self::Error> {
        match src {
            AssignOp::Basic => Err(()),
            AssignOp::Plus => Ok(Self::Add),
            AssignOp::Minus => Ok(Self::Sub),
            AssignOp::Mul => Ok(Self::Mul),
            AssignOp::Div => Ok(Self::Div),
            AssignOp::Rem => Ok(Self::Rem),
            AssignOp::Pow => Ok(Self::Pow),
            AssignOp::Shl => Ok(Self::Shl),
            AssignOp::Shr => Ok(Self::Shr),
            AssignOp::BitOr => Ok(Self::BitOr),
            AssignOp::BitAnd => Ok(Self::BitAnd),
            AssignOp::BitXor => Ok(Self::BitXor),
        }
    }
}

impl From<PrefixOp> for Instruction {
    fn from(src: PrefixOp) -> Self {
        match src {
            PrefixOp::Not => Self::Not,
            PrefixOp::Neg => Self::Neg,
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Push(val) => write!(f, "push {val}"),
            Instruction::Drop => write!(f, "pop"),
            Instruction::Jmp(idx) => write!(f, "jmp {idx}"),
            Instruction::JmpFalse(idx) => write!(f, "jmpfalse {idx}"),
            Instruction::SetVar(idx) => write!(f, "setvar {idx}"),
            Instruction::GetVar(idx) => write!(f, "getvar {idx}"),
            Instruction::SetGlobal(idx) => write!(f, "setglob {idx}"),
            Instruction::GetGlobal(idx) => write!(f, "getglob {idx}"),
            Instruction::Call(idx) => write!(f, "call {idx}"),
            Instruction::Cast(to) => write!(f, "cast {to}"),
            Instruction::Ret => write!(f, "ret"),
            Instruction::Exit => write!(f, "exit"),
            Instruction::Not => write!(f, "not"),
            Instruction::Neg => write!(f, "neg"),
            Instruction::Add => write!(f, "add"),
            Instruction::Sub => write!(f, "sub"),
            Instruction::Mul => write!(f, "mul"),
            Instruction::Pow => write!(f, "pow"),
            Instruction::Div => write!(f, "div"),
            Instruction::Rem => write!(f, "rem"),
            Instruction::Eq => write!(f, "eq"),
            Instruction::Ne => write!(f, "ne"),
            Instruction::Lt => write!(f, "lt"),
            Instruction::Gt => write!(f, "gt"),
            Instruction::Le => write!(f, "le"),
            Instruction::Ge => write!(f, "ge"),
            Instruction::Shl => write!(f, "shl"),
            Instruction::Shr => write!(f, "shr"),
            Instruction::BitOr => write!(f, "bitor"),
            Instruction::BitAnd => write!(f, "bitand"),
            Instruction::BitXor => write!(f, "bitxor"),
        }
    }
}

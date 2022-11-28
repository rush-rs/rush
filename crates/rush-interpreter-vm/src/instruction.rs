use std::fmt::Display;

use rush_analyzer::{AssignOp, InfixOp, PrefixOp, Type as AnalyzerType};

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

use crate::Value;

#[derive(Debug)]
pub enum Instruction {
    /// Adds a new constant to stack.
    Push(Value),
    Pop,
    // Calls a function (specified by index).
    Call(usize),
    /// Returns from the current function call.
    Ret,
    /// Special instruction for exit calls.
    Exit,
    /// Jumps to the specified index.
    Jmp(usize),
    /// Jumps to the specified index if the value on the stack is `false`.
    JmpCond(usize),
    /// Pops the top element off the stack and binds it to the specified number.
    SetVar(usize),
    /// Retrieves the variable with the specified index and places it on top of the stack.
    GetVar(usize),
    // Pops the top element off the stack and sets it as a global using the specified index.
    SetGlob(usize),
    /// Retrieves the global with the specified index and places it on top of the stack.
    GetGlob(usize),
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
    And,
    Or,
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
            InfixOp::And => Self::And,
            InfixOp::Or => Self::Or,
        }
    }
}

impl From<AssignOp> for Instruction {
    fn from(src: AssignOp) -> Self {
        match src {
            AssignOp::Basic => unreachable!("never called using this operator"),
            AssignOp::Plus => Self::Add,
            AssignOp::Minus => Self::Sub,
            AssignOp::Mul => Self::Mul,
            AssignOp::Div => Self::Div,
            AssignOp::Rem => Self::Rem,
            AssignOp::Pow => Self::Pow,
            AssignOp::Shl => Self::Shl,
            AssignOp::Shr => Self::Shr,
            AssignOp::BitOr => Self::BitOr,
            AssignOp::BitAnd => Self::BitAnd,
            AssignOp::BitXor => Self::BitXor,
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Push(val) => write!(f, "push {val}"),
            Instruction::Pop => write!(f, "pop"),
            Instruction::Jmp(idx) => write!(f, "jmp {idx}"),
            Instruction::JmpCond(idx) => write!(f, "jmpcond {idx}"),
            Instruction::SetVar(idx) => write!(f, "setvar {idx}"),
            Instruction::GetVar(idx) => write!(f, "getvar {idx}"),
            Instruction::SetGlob(idx) => write!(f, "setglob {idx}"),
            Instruction::GetGlob(idx) => write!(f, "getglob {idx}"),
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
            Instruction::And => write!(f, "and"),
            Instruction::Or => write!(f, "or"),
        }
    }
}

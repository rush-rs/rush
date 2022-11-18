#![allow(dead_code)] // TODO: remove this attribute

use std::fmt::{self, Display, Formatter};

use crate::register::{FloatRegister, IntRegister};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Int(IntValue),
    IntOrImm(IntValueOrImm),
    Float(FloatValue),
}

impl Value {
    pub fn unwrap_int(self) -> IntValue {
        match self {
            Self::Int(int) => int,
            Self::IntOrImm(int_or_imm) => match int_or_imm.try_into() {
                Ok(int) => int,
                Err(_) => panic!("called `unwrap_int` on non-int variant"),
            },
            _ => panic!("called `unwrap_int` on non-int variant"),
        }
    }

    pub fn unwrap_int_or_imm(self) -> IntValueOrImm {
        match self {
            Self::Int(int) => int.into(),
            Self::IntOrImm(int_or_imm) => int_or_imm,
            _ => panic!("called `unwrap_int_or_imm` on float variant"),
        }
    }

    pub fn unwrap_float(self) -> FloatValue {
        match self {
            Self::Float(float) => float,
            _ => panic!("called `unwrap_float` on non-float variant"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FloatValue {
    Register(FloatRegister),
    Ptr(Size, IntRegister, Offset),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntValue {
    Register(IntRegister),
    Ptr(Size, IntRegister, Offset),
}

impl TryFrom<IntValueOrImm> for IntValue {
    type Error = ();

    fn try_from(value: IntValueOrImm) -> Result<Self, Self::Error> {
        match value {
            IntValueOrImm::Register(reg) => Ok(Self::Register(reg)),
            IntValueOrImm::Ptr(size, reg, offset) => Ok(Self::Ptr(size, reg, offset)),
            IntValueOrImm::Immediate(_) => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntValueOrImm {
    Register(IntRegister),
    Ptr(Size, IntRegister, Offset),
    Immediate(i64),
}

impl From<IntValue> for IntValueOrImm {
    fn from(value: IntValue) -> Self {
        match value {
            IntValue::Register(reg) => IntValueOrImm::Register(reg),
            IntValue::Ptr(size, reg, offset) => IntValueOrImm::Ptr(size, reg, offset),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Size {
    Byte,
    Word,
    Dword,
    Qword,
    Oword,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Offset {
    Immediate(i64),
    Symbol(String),
}

/////////////////////////////////////////////////

impl Display for FloatValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            FloatValue::Register(reg) => write!(f, "{reg}"),
            FloatValue::Ptr(size, reg, offset) => write!(f, "{size} ptr [{reg} + {offset}]"),
        }
    }
}

impl Display for IntValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            IntValue::Register(reg) => write!(f, "{reg}"),
            IntValue::Ptr(size, reg, offset) => write!(f, "{size} ptr [{reg} + {offset}]"),
        }
    }
}

impl Display for IntValueOrImm {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            IntValueOrImm::Register(reg) => write!(f, "{reg}"),
            IntValueOrImm::Ptr(size, reg, offset) => write!(f, "{size} ptr [{reg} + {offset}]"),
            IntValueOrImm::Immediate(num) => write!(f, "{num}"),
        }
    }
}

impl Display for Size {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Size::Byte => "byte",
                Size::Word => "word",
                Size::Dword => "dword",
                Size::Qword => "qword",
                Size::Oword => "oword",
            }
        )
    }
}

impl Display for Offset {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Offset::Immediate(num) => write!(f, "{num}"),
            Offset::Symbol(symbol) => write!(f, "{symbol}"),
        }
    }
}

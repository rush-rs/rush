#![allow(dead_code)] // TODO: remove this attribute

use std::{
    cmp::Ordering,
    fmt::{self, Display, Formatter},
};

use rush_analyzer::Type;

use crate::register::{FloatRegister, IntRegister};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Int(IntValue),
    Float(FloatValue),
}

impl Value {
    pub fn unwrap_int(self) -> IntValue {
        match self {
            Self::Int(int) => int,
            _ => panic!("called `unwrap_int` on non-int variant"),
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
    Ptr(Pointer),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntValue {
    Register(IntRegister),
    Ptr(Pointer),
    Immediate(i64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pointer {
    pub size: Size,
    pub base: IntRegister,
    pub offset: Offset,
}

impl Pointer {
    pub fn new(size: Size, base: IntRegister, offset: Offset) -> Self {
        Self { size, base, offset }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Size {
    Byte = 1,
    Word = 2,
    Dword = 4,
    Qword = 8,
    Oword = 16,
}

impl Size {
    pub fn byte_count(&self) -> i64 {
        *self as i64
    }

    // TODO: remove bit_count method?
    pub fn bit_count(&self) -> i64 {
        self.byte_count() * 8
    }

    pub fn min_for_value(value: &i64) -> Self {
        // ranges entered manually, because primitive constants aren't i64
        match *value {
            // i8::MIN..=u8::MAX
            -0x80..=0xFF => Size::Byte,
            // i16::MIN..=u16::MAX
            -0x8000..=0xFFFF => Size::Word,
            // i32::MIN..=u32::MAX
            -0x8000_0000..=0xFFFF_FFFF => Size::Dword,
            // i64::MIN..=i64::MAX
            -0x8000_0000_0000_0000..=0x7FFF_FFFF_FFFF_FFFF => Size::Qword,
        }
    }
}

impl TryFrom<Type> for Size {
    type Error = ();

    fn try_from(value: Type) -> Result<Self, Self::Error> {
        match value {
            Type::Int | Type::Float => Ok(Size::Qword),
            Type::Bool | Type::Char => Ok(Size::Byte),
            Type::Unit | Type::Never => Err(()),
            Type::Unknown => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
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
            FloatValue::Ptr(ptr) => write!(f, "{ptr}"),
        }
    }
}

impl Display for IntValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            IntValue::Register(reg) => write!(f, "{reg}"),
            IntValue::Ptr(ptr) => write!(f, "{ptr}"),
            IntValue::Immediate(num) => write!(f, "{num}"),
        }
    }
}

impl Display for Pointer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pointer { size, base, offset } = self;
        write!(f, "{size} ptr [{base}{offset}]")
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
            Offset::Immediate(num) => match num.cmp(&0) {
                Ordering::Less => write!(f, "{num}"),
                Ordering::Equal => Ok(()),
                Ordering::Greater => write!(f, "+{num}"),
            },
            Offset::Symbol(symbol) => write!(f, "+{symbol}"),
        }
    }
}

/////////////////////////////////////////////////

impl From<FloatRegister> for FloatValue {
    fn from(reg: FloatRegister) -> Self {
        Self::Register(reg)
    }
}

impl From<Pointer> for FloatValue {
    fn from(ptr: Pointer) -> Self {
        Self::Ptr(ptr)
    }
}

impl From<IntRegister> for IntValue {
    fn from(reg: IntRegister) -> Self {
        Self::Register(reg)
    }
}

impl From<Pointer> for IntValue {
    fn from(ptr: Pointer) -> Self {
        Self::Ptr(ptr)
    }
}

impl From<i64> for IntValue {
    fn from(num: i64) -> Self {
        Self::Immediate(num)
    }
}

impl From<i64> for Offset {
    fn from(num: i64) -> Self {
        Self::Immediate(num)
    }
}

impl From<String> for Offset {
    fn from(symbol: String) -> Self {
        Self::Symbol(symbol)
    }
}

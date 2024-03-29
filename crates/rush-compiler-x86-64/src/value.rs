use std::{
    cmp::Ordering,
    fmt::{self, Display, Formatter},
    rc::Rc,
};

use rush_analyzer::Type;

use crate::register::{FloatRegister, IntRegister, Register};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Int(IntValue),
    Float(FloatValue),
}

impl Value {
    pub fn expect_int(self, msg: impl Display) -> IntValue {
        match self {
            Self::Int(int) => int,
            _ => panic!("called `expect_int` on non-int variant: {msg}"),
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

    pub fn in_size(self, size: Size) -> Self {
        Self { size, ..self }
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

    pub fn mask(&self) -> i64 {
        match self {
            Size::Byte => 0xff,
            Size::Word => 0xffff,
            Size::Dword => 0xffff_ffff,
            Size::Qword => -1,
            Size::Oword => panic!("oword mask too large for 64 bits"),
        }
    }
}

impl TryFrom<Type> for Size {
    type Error = ();

    fn try_from(value: Type) -> Result<Self, Self::Error> {
        match value {
            Type::Int(0) | Type::Float(0) => Ok(Size::Qword),
            Type::Bool(0) | Type::Char(0) => Ok(Size::Byte),
            // pointers are 64-bit on x86_64
            Type::Int(_) | Type::Float(_) | Type::Bool(_) | Type::Char(_) => Ok(Size::Qword),
            Type::Unit | Type::Never => Err(()),
            Type::Unknown => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Offset {
    Immediate(i64),
    Label(Rc<str>),
}

/////////////////////////////////////////////////

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(int) => write!(f, "{int}"),
            Value::Float(float) => write!(f, "{float}"),
        }
    }
}

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
            match f.alternate() {
                true => match self {
                    Size::Byte => "byte",
                    Size::Word => "short",
                    Size::Dword => "long",
                    Size::Qword => "quad",
                    Size::Oword => "octa",
                },
                false => match self {
                    Size::Byte => "byte",
                    Size::Word => "word",
                    Size::Dword => "dword",
                    Size::Qword => "qword",
                    Size::Oword => "oword",
                },
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
            Offset::Label(label) => write!(f, "+{label}"),
        }
    }
}

/////////////////////////////////////////////////

impl From<Register> for Value {
    fn from(reg: Register) -> Self {
        match reg {
            Register::Int(reg) => Self::Int(IntValue::Register(reg)),
            Register::Float(reg) => Self::Float(FloatValue::Register(reg)),
        }
    }
}

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

impl From<Rc<str>> for Offset {
    fn from(label: Rc<str>) -> Self {
        Self::Label(label)
    }
}

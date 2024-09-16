use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Register {
    Int(IntRegister),
    Float(FloatRegister),
}

impl Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Register::Int(reg) => reg.to_string(),
                Register::Float(reg) => reg.to_string(),
            }
        )
    }
}

impl From<IntRegister> for Register {
    fn from(src: IntRegister) -> Self {
        Self::Int(src)
    }
}

impl From<FloatRegister> for Register {
    fn from(src: FloatRegister) -> Self {
        Self::Float(src)
    }
}

pub(crate) const INT_REGISTERS: &[IntRegister] = &[
    IntRegister::R0,
    IntRegister::R1,
    IntRegister::R2,
    IntRegister::R3,
    IntRegister::R4,
    IntRegister::R5,
    IntRegister::R6,
    IntRegister::R7,
    IntRegister::R8,
    IntRegister::R9,
    IntRegister::R10,
    IntRegister::R11,
    IntRegister::R12,
    IntRegister::R13,
    IntRegister::R14,
    // Excluded: must never be used.
    // IntRegister::GR15,
];

pub(crate) const FLOAT_REGISTERS: &[FloatRegister] = &[
    FloatRegister::F0,
    FloatRegister::F1,
    FloatRegister::F2,
    FloatRegister::F3,
    FloatRegister::F4,
    FloatRegister::F5,
    FloatRegister::F6,
    FloatRegister::F7,
    FloatRegister::F8,
    FloatRegister::F9,
    FloatRegister::F10,
    FloatRegister::F11,
    FloatRegister::F12,
    FloatRegister::F13,
    FloatRegister::F14,
    FloatRegister::F15,
];

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum IntRegister {
    // special.
    R15, // (stack pointer)
    // other.
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
}

impl IntRegister {
    pub(crate) fn nth_param(n: usize) -> Option<Self> {
        [
            Self::R2,
            Self::R3,
            Self::R4,
            Self::R5,
            Self::R6,
            Self::R7,
            Self::R8,
            Self::R9,
        ]
        .into_iter()
        .nth(n)
    }

    #[inline]
    pub(crate) fn to_reg(self) -> Register {
        Register::Int(self)
    }
}

impl From<Register> for IntRegister {
    fn from(src: Register) -> Self {
        match src {
            Register::Int(reg) => reg,
            Register::Float(_) => panic!("cannot convert float register into int register"),
        }
    }
}

impl Display for IntRegister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // lowercase debug display
        write!(f, "{}", format!("%{self:?}").to_lowercase())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FloatRegister {
    // general
    F0,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
}

impl FloatRegister {
    pub(crate) fn nth_param(n: usize) -> Option<Self> {
        [
            Self::F0,
            Self::F1,
            Self::F2,
            Self::F3,
            Self::F4,
            Self::F5,
            Self::F6,
            Self::F7,
        ]
        .into_iter()
        .nth(n)
    }

    #[inline]
    pub(crate) fn to_reg(self) -> Register {
        Register::Float(self)
    }
}

impl From<Register> for FloatRegister {
    fn from(src: Register) -> Self {
        match src {
            Register::Float(reg) => reg,
            Register::Int(_) => panic!("cannot convert int register into float register"),
        }
    }
}

impl Display for FloatRegister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // lowercase debug display
        write!(f, "{}", format!("%{self:?}").to_lowercase())
    }
}

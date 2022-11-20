#![allow(dead_code)] // TODO: remove this attribute

use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Register {
    Int(IntRegister),
    Float(FloatRegister),
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
    IntRegister::A0,
    IntRegister::A1,
    IntRegister::A2,
    IntRegister::A3,
    IntRegister::A4,
    IntRegister::A5,
    IntRegister::A6,
    IntRegister::A7,
    IntRegister::T0,
    IntRegister::T1,
    IntRegister::T2,
    IntRegister::T3,
    IntRegister::T4,
    IntRegister::T5,
    IntRegister::S1,
    IntRegister::S2,
    IntRegister::S3,
    IntRegister::S4,
    IntRegister::S5,
    IntRegister::S6,
    IntRegister::S7,
    IntRegister::S8,
    IntRegister::S9,
    IntRegister::S10,
    IntRegister::S11,
];

pub(crate) const FLOAT_REGISTERS: &[FloatRegister] = &[
    FloatRegister::Fa0,
    FloatRegister::Fa1,
    FloatRegister::Fa2,
    FloatRegister::Fa3,
    FloatRegister::Fa4,
    FloatRegister::Fa5,
    FloatRegister::Fa6,
    FloatRegister::Fa7,
    FloatRegister::Ft0,
    FloatRegister::Ft1,
    FloatRegister::Ft2,
    FloatRegister::Ft3,
    FloatRegister::Ft4,
    FloatRegister::Ft5,
    FloatRegister::Ft6,
    FloatRegister::Ft7,
    FloatRegister::Ft8,
    FloatRegister::Ft9,
    FloatRegister::Ft10,
    FloatRegister::Ft11,
    FloatRegister::Fs0,
    FloatRegister::Fs1,
    FloatRegister::Fs2,
    FloatRegister::Fs3,
    FloatRegister::Fs4,
    FloatRegister::Fs5,
    FloatRegister::Fs6,
    FloatRegister::Fs7,
    FloatRegister::Fs8,
    FloatRegister::Fs9,
    FloatRegister::Fs10,
    FloatRegister::Fs11,
];

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum IntRegister {
    // special
    Zero,
    Ra,
    // pointers
    Sp,
    Gp,
    Tp,
    Fp,
    // temporaries
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    // T6: DISABLED: used during instruction code generation,
    // saved
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    // args
    A0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
}

impl IntRegister {
    pub(crate) fn nth_param(n: usize) -> Option<Self> {
        Some(match n {
            0 => Self::A0,
            1 => Self::A1,
            2 => Self::A2,
            3 => Self::A3,
            4 => Self::A4,
            5 => Self::A5,
            6 => Self::A6,
            7 => Self::A7,
            _ => return None,
        })
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
        write!(f, "{}", format!("{self:?}").to_lowercase())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FloatRegister {
    // temporaries
    Ft0,
    Ft1,
    Ft2,
    Ft3,
    Ft4,
    Ft5,
    Ft6,
    Ft7,
    Ft8,
    Ft9,
    Ft10,
    Ft11,
    // saved
    Fs0,
    Fs1,
    Fs2,
    Fs3,
    Fs4,
    Fs5,
    Fs6,
    Fs7,
    Fs8,
    Fs9,
    Fs10,
    Fs11,
    // args
    Fa0,
    Fa1,
    Fa2,
    Fa3,
    Fa4,
    Fa5,
    Fa6,
    Fa7,
}

impl FloatRegister {
    pub(crate) fn nth_param(n: usize) -> Option<Self> {
        Some(match n {
            0 => Self::Fa0,
            1 => Self::Fa1,
            2 => Self::Fa2,
            3 => Self::Fa3,
            4 => Self::Fa4,
            5 => Self::Fa5,
            6 => Self::Fa6,
            7 => Self::Fa7,
            _ => return None,
        })
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
        write!(f, "{}", format!("{self:?}").to_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_display() {
        let tests = [
            (IntRegister::Zero, "zero"),
            (IntRegister::A0, "a0"),
            (IntRegister::Ra, "ra"),
            (IntRegister::Fp, "fp"),
            (IntRegister::T0, "t0"),
            (IntRegister::S6, "s6"),
        ];

        for (reg, display) in tests {
            assert_eq!(reg.to_string(), display);
        }
    }

    #[test]
    fn test_float_register_display() {
        let tests = [
            (FloatRegister::Ft0, "ft0"),
            (FloatRegister::Fs1, "fs1"),
            (FloatRegister::Fa2, "fa2"),
            (FloatRegister::Fs5, "fs5"),
        ];

        for (reg, display) in tests {
            assert_eq!(reg.to_string(), display);
        }
    }
}

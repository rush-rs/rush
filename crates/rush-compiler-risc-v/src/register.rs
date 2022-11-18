#![allow(dead_code)] // TODO: remove this attribute

use std::fmt::Display;

#[derive(Debug)]
pub enum Register {
    Int(IntRegister),
    Float(FloatRegister),
}

#[derive(Debug)]
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

impl Display for IntRegister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // lowercase debug display
        write!(f, "{}", format!("{self:?}").to_lowercase())
    }
}

#[derive(Debug)]
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

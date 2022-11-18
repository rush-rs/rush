#![allow(dead_code)] // TODO: remove this attribute

use std::fmt::Display;

use crate::register::{FloatRegister, Register};

pub enum Instruction {
    Ret,
    Ecall,

    Jmp(String),
    BrCond(Condition, Register, Register, String),

    // base integer instructions
    SetIntCondition(Condition, Register, Register, Register),
    Snez(Register, Register),
    Li(Register, i64),
    Add(Register, Register, Register),
    Sub(Register, Register, Register),
    Mul(Register, Register, Register),
    Div(Register, Register, Register),
    Rem(Register, Register, Register),
    Xor(Register, Register, Register),
    Or(Register, Register, Register),
    And(Register, Register, Register),
    Sl(Register, Register, Register),
    Sr(Register, Register, Register),

    // load / store operations
    Lb(Register, Pointer),
    Ld(Register, Pointer),
    Sb(Register, Pointer),
    Sd(Register, Pointer),

    // floats (arithmetic instructions use `.d` suffix)
    SetFloatCondition(Condition, FloatRegister, FloatRegister, FloatRegister),
    Fld(FldType),
    Fadd(FloatRegister, FloatRegister, FloatRegister),
    Fsub(FloatRegister, FloatRegister, FloatRegister),
    Fmul(FloatRegister, FloatRegister, FloatRegister),
    Fdiv(FloatRegister, FloatRegister, FloatRegister),

    // casts
    CastIntToFloat(FloatRegister, Register),
    CastFloatToInt(Register, FloatRegister),
    CastByteToFloat(FloatRegister, Register),
}

pub enum FldType {
    Stack(FloatRegister, Register, i64),
    /// Requires additional temp register at the end (fld ft0, test, t0)
    Label(FloatRegister, String, Register),
}

impl Display for FldType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stack(dest, reg, offset) => write!(f, "fld {dest}, {offset}({reg})"),
            Self::Label(dest, label, tmp_reg) => write!(f, "fld {dest}, {label}, {tmp_reg}"),
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Ret => write!(f, "ret"),
            Instruction::Ecall => write!(f, "ecall"),
            Instruction::Jmp(label) => write!(f, "j {label}"),
            Instruction::BrCond(cond, l, r, lbl) => write!(f, "b{cond} {l}, {r}, {lbl}"),
            Instruction::SetIntCondition(cond, dest, l, r) => match cond {
                Condition::Lt => write!(f, "slt {dest}, {l}, {r}"),
                // Because RISC-V does not support the sle instruction, it is emulated here
                Condition::Le => {
                    writeln!(f, "# begin sle")?;
                    writeln!(f, "xor t6, {l}, {r}")?;
                    writeln!(f, "seqz t6, t6")?;
                    writeln!(f, "slt {dest}, {l}, {r}")?;
                    writeln!(f, "or {dest}, {dest}, t6")?;
                    write!(f, "# end sle")
                }
                // Because RISC-V does not support the seq instruction, it is emulated here
                Condition::Eq => {
                    writeln!(f, "# begin seq")?;
                    writeln!(f, "xor {dest}, {l}, {r}")?;
                    writeln!(f, "seqz {dest}, {dest}")?;
                    write!(f, "# end seq")
                }
                // Because RISC-V does not support the sne instruction, it is emulated here
                Condition::Ne => {
                    writeln!(f, "# begin sne")?;
                    writeln!(f, "xor {dest}, {l}, {r}")?;
                    writeln!(f, "snez {dest}, {dest}")?;
                    write!(f, "# end sne")
                }
                Condition::Gt => write!(f, "sgt {dest}, {l}, {r}"),
                Condition::Ge => {
                    writeln!(f, "# begin sge")?;
                    writeln!(f, "xor t6, {l}, {r}")?;
                    writeln!(f, "seqz t6, t6")?;
                    writeln!(f, "sgt {dest}, {l}, {r}")?;
                    writeln!(f, "or {dest}, {dest}, t6")?;
                    write!(f, "# end sge")
                }
            },
            Instruction::Li(dest, val) => write!(f, "li {dest}, {val}"),
            Instruction::Add(dest, lhs, rhs) => write!(f, "add {dest}, {lhs}, {rhs}"),
            Instruction::Sub(dest, lhs, rhs) => write!(f, "sub {dest}, {lhs}, {rhs}"),
            Instruction::Mul(dest, lhs, rhs) => write!(f, "mul {dest}, {lhs}, {rhs}"),
            Instruction::Div(dest, lhs, rhs) => write!(f, "div {dest}, {lhs}, {rhs}"),
            Instruction::Rem(dest, lhs, rhs) => write!(f, "rem {dest}, {lhs}, {rhs}"),
            Instruction::Xor(dest, lhs, rhs) => write!(f, "xor {dest}, {lhs}, {rhs}"),
            Instruction::Or(dest, lhs, rhs) => write!(f, "or {dest}, {lhs}, {rhs}"),
            Instruction::And(dest, lhs, rhs) => write!(f, "and {dest}, {lhs}, {rhs}"),
            Instruction::Sl(dest, lhs, rhs) => write!(f, "sll {dest}, {lhs}, {rhs}"),
            Instruction::Sr(dest, lhs, rhs) => write!(f, "sra {dest}, {lhs},{rhs}"),
            Instruction::Lb(dest, ptr) => write!(f, "lb {dest}, {ptr}"),
            Instruction::Ld(dest, ptr) => write!(f, "ld {dest}, {ptr}"),
            Instruction::Sb(dest, ptr) => write!(f, "sb {dest}, {ptr}"),
            Instruction::Sd(dest, ptr) => write!(f, "sd {dest}, {ptr}"),
            Instruction::Fld(type_) => write!(f, "{type_}"),
            Instruction::Fadd(dest, lhs, rhs) => write!(f, "fadd.d {dest}, {lhs}, {rhs}"),
            Instruction::Fsub(dest, lhs, rhs) => write!(f, "fsub.d {dest}, {lhs}, {rhs}"),
            Instruction::Fmul(dest, lhs, rhs) => write!(f, "fmul.d {dest}, {lhs}, {rhs}"),
            Instruction::Fdiv(dest, lhs, rhs) => write!(f, "fdiv.d {dest}, {lhs}, {rhs}"),
            Instruction::SetFloatCondition(cond, dest, l, r) => match cond {
                Condition::Lt => write!(f, "flt.d {dest}, {l}, {r}"),
                Condition::Le => write!(f, "file.d {dest}, {l}, {r}"),
                Condition::Eq => write!(f, "feq.d {dest}, {l}, {r}"),
                Condition::Ne => {
                    writeln!(f, "feq.d {dest}, {l}, {r}")?;
                    write!(f, "seqz {dest}, {dest}")
                }
                Condition::Gt => write!(f, "fgt.d {dest}, {l}, {r}"),
                Condition::Ge => write!(f, "fge.d {dest}, {l}, {r}"),
            },
        }
    }
}

pub enum Condition {
    Lt,
    Le,
    Eq,
    Ne,
    Gt,
    Ge,
}

impl Display for Condition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Lt => "lt",
                Self::Le => "le",
                Self::Gt => "gt",
                Self::Ge => "ge",
                Self::Eq => "eq",
                Self::Ne => "ne",
            }
        )
    }
}

pub enum Pointer {
    Stack(Register, i64),
    Label(String),
}

impl Display for Pointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stack(reg, offset) => write!(f, "{}({})", offset, reg),
            Self::Label(label) => write!(f, "{label}"),
        }
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_display() {
        let tests = [
            (Instruction::Ret, "ret"),
            (
                Instruction::Fadd(FloatRegister::Ft0, FloatRegister::Ft1, FloatRegister::Ft3),
                "fadd.d ft0, ft1, ft3",
            ),
            (
                Instruction::Add(Register::T3, Register::S5, Register::A0),
                "add t3, s5, a0",
            ),
            (Instruction::Li(Register::A0, 1), "li a0, 1"),
            (
                Instruction::Ld(Register::T0, Pointer::Stack(Register::Sp, 16)),
                "ld t0, 16(sp)",
            ),
            (
                Instruction::Lb(Register::T0, Pointer::Label("int".to_string())),
                "lb t0, int",
            ),
            (
                Instruction::Fld(FldType::Label(
                    FloatRegister::Ft0,
                    "a".to_string(),
                    Register::T0,
                )),
                "fld ft0, a, t0",
            ),
            (
                Instruction::Fld(FldType::Stack(FloatRegister::Ft0, Register::Sp, 32)),
                "fld ft0, 32(sp)",
            ),
            (
                Instruction::SetIntCondition(
                    Condition::Le,
                    Register::A0,
                    Register::A0,
                    Register::A1,
                ),
                "# begin sle\nxor t6, a0, a1\nseqz t6, t6\nslt a0, a0, a1\nor a0, a0, t6\n# end sle",
            ),
            (
                Instruction::SetIntCondition(
                    Condition::Ge,
                    Register::A0,
                    Register::A0,
                    Register::A1,
                ),
                "# begin sge\nxor t6, a0, a1\nseqz t6, t6\nsgt a0, a0, a1\nor a0, a0, t6\n# end sge",
            ),
        ];

        for (inst, display) in tests {
            println!("{inst}");
            assert_eq!(inst.to_string(), display)
        }
    }
}

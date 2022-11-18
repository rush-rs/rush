#![allow(dead_code)] // TODO: remove this attribute

use std::fmt::{self, Display, Formatter};

use crate::register::{FloatRegister, IntRegister};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    IntelSyntax,
    Global(String),
    Section(Section),
    Symbol(String),
    Byte(u8),
    Short(u16),
    Long(u32),
    Quad(u64),
    Octa(u128),

    Ret,
    Syscall,

    Jmp(String),
    JCond(Condition, String),
    SetCond(Condition, IntRegister),

    //////////////////////////
    //////// Integers ////////
    //////////////////////////
    Mov(IntValue, IntValueOrImm),

    Add(IntValue, IntValueOrImm),
    Sub(IntValue, IntValueOrImm),
    Imul(IntValue, IntValueOrImm),
    Idiv(IntValue),

    Inc(IntValue),
    Dec(IntValue),
    Neg(IntValue),

    /// Logical shift left
    Shl(IntValue, IntValueOrImm),
    /// Arithmetic shift right
    Sar(IntValue, IntValueOrImm),

    And(IntValue, IntValueOrImm),
    Or(IntValue, IntValueOrImm),
    Xor(IntValue, IntValueOrImm),

    Cmp(IntValue, IntValueOrImm),
    /// Convert int to float. (**c**on**v**er**t** **s**igned **i**nteger to **s**calar **d**ouble)
    Cvtsi2sd(FloatValue, IntValue),

    ////////////////////////
    //////// Floats ////////
    ////////////////////////
    /// Move scalar double
    Movsd(FloatValue, FloatValue),

    /// Add scalar double
    Addsd(FloatValue, FloatValue),
    /// Subtract scalar double
    Subsd(FloatValue, FloatValue),
    /// Multiply scalar double
    Mulsd(FloatValue, FloatValue),
    /// Divide scalar double
    Divsd(FloatValue, FloatValue),

    /// **Xor** **p**acked **d**ouble
    Xorpd(FloatValue, FloatValue),

    /// Compare floats. (**u**nordered **com**pare **s**calar **d**ouble)
    Ucomisd(FloatValue, FloatValue),
    /// Convert float to int. (**c**on**v**er**t** **t**runcating **s**calar **d**ouble to
    /// **s**igned **i**nteger)
    Cvttsd2si(IntValue, FloatValue),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Section {
    Text,
    Data,
    ReadOnlyData,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntValueOrImm {
    Register(IntRegister),
    Ptr(Size, IntRegister, Offset),
    Immediate(i64),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    Above,
    AboveOrEqual,
    Below,
    BelowOrEqual,

    Equal,
    NotEqual,

    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
}

/////////////////////////////////////////////////

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::IntelSyntax => writeln!(f, ".intel_syntax"),
            Instruction::Global(name) => writeln!(f, ".global {name}"),
            Instruction::Section(section) => writeln!(f, ".section {section}"),
            Instruction::Symbol(name) => writeln!(f, "{name}:"),
            Instruction::Byte(num) => writeln!(f, "    .byte {num:#04x}"),
            Instruction::Short(num) => writeln!(f, "    .short {num:#06x}"),
            Instruction::Long(num) => writeln!(f, "    .long {num:#010x}"),
            Instruction::Quad(num) => writeln!(f, "    .quad {num:#018x}"),
            Instruction::Octa(num) => writeln!(f, "    .octa {num:#034x}"),
            Instruction::Ret => writeln!(f, "    ret"),
            Instruction::Syscall => writeln!(f, "    syscall"),
            Instruction::Jmp(symbol) => writeln!(f, "    jmp {symbol}"),
            Instruction::JCond(cond, symbol) => writeln!(f, "    j{cond} {symbol}"),
            Instruction::SetCond(cond, reg) => writeln!(f, "    set{cond} {reg}"),
            Instruction::Mov(dest, src) => writeln!(f, "    mov {dest}, {src}"),
            Instruction::Add(dest, src) => writeln!(f, "    add {dest}, {src}"),
            Instruction::Sub(dest, src) => writeln!(f, "    sub {dest}, {src}"),
            Instruction::Imul(dest, src) => writeln!(f, "    imul {dest}, {src}"),
            Instruction::Idiv(divisor) => writeln!(f, "    idiv {divisor}"),
            Instruction::Inc(reg) => writeln!(f, "    inc {reg}"),
            Instruction::Dec(reg) => writeln!(f, "    dec {reg}"),
            Instruction::Neg(reg) => writeln!(f, "    neg {reg}"),
            Instruction::Shl(dest, src) => writeln!(f, "    shl {dest}, {src}"),
            Instruction::Sar(dest, src) => writeln!(f, "    sar {dest}, {src}"),
            Instruction::And(dest, src) => writeln!(f, "    and {dest}, {src}"),
            Instruction::Or(dest, src) => writeln!(f, "    or {dest}, {src}"),
            Instruction::Xor(dest, src) => writeln!(f, "    xor {dest}, {src}"),
            Instruction::Cmp(left, right) => writeln!(f, "    cmp {left}, {right}"),
            Instruction::Cvtsi2sd(dest, src) => writeln!(f, "    cvtsi2sd {dest}, {src}"),
            Instruction::Movsd(dest, src) => writeln!(f, "    movsd {dest}, {src}"),
            Instruction::Addsd(dest, src) => writeln!(f, "    addsd {dest}, {src}"),
            Instruction::Subsd(dest, src) => writeln!(f, "    subsd {dest}, {src}"),
            Instruction::Mulsd(dest, src) => writeln!(f, "    mulsd {dest}, {src}"),
            Instruction::Divsd(dest, src) => writeln!(f, "    divsd {dest}, {src}"),
            Instruction::Xorpd(dest, src) => writeln!(f, "    xorpd {dest}, {src}"),
            Instruction::Ucomisd(left, right) => writeln!(f, "    ucomisd {left}, {right}"),
            Instruction::Cvttsd2si(dest, src) => writeln!(f, "    cvttsd2si {dest}, {src}"),
        }
    }
}

impl Display for Section {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            ".{}",
            match self {
                Section::Text => "text",
                Section::Data => "data",
                Section::ReadOnlyData => "rodata",
            }
        )
    }
}

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

impl Display for Condition {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Condition::Above => "a",
                Condition::AboveOrEqual => "ae",
                Condition::Below => "b",
                Condition::BelowOrEqual => "be",
                Condition::Equal => "e",
                Condition::NotEqual => "ne",
                Condition::Greater => "g",
                Condition::GreaterOrEqual => "ge",
                Condition::Less => "l",
                Condition::LessOrEqual => "le",
            }
        )
    }
}

/////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let instrs = [
            Instruction::Cmp(
                IntValue::Register(IntRegister::Rsi),
                IntValueOrImm::Immediate(0),
            ),
            Instruction::JCond(Condition::Less, "return_0".to_string()),
            Instruction::Mov(
                IntValue::Register(IntRegister::Rax),
                IntValueOrImm::Immediate(1),
            ),
            Instruction::Ucomisd(
                FloatValue::Register(FloatRegister::Xmm0),
                FloatValue::Ptr(
                    Size::Qword,
                    IntRegister::Rip,
                    Offset::Symbol("float_0".to_string()),
                ),
            ),
            Instruction::Section(Section::ReadOnlyData),
            Instruction::Symbol("float_127".to_string()),
            Instruction::Quad(127_f64.to_bits()),
        ];
        for instr in instrs {
            print!("{instr}");
        }
    }
}

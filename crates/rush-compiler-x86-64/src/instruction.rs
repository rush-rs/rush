#![allow(dead_code)] // TODO: remove this attribute

use std::fmt::{self, Display, Formatter};

use crate::{
    register::IntRegister,
    value::{FloatValue, IntValue},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    IntelSyntax,
    Global(String),
    Section(Section),
    Symbol(String),
    Byte(u8),
    Short(u16),
    Long(u32),
    QuadInt(u64),
    QuadFloat(f64),
    Octa(u128),

    Leave,
    Ret,
    Syscall,
    Call(String),
    Push(IntRegister),
    Pop(IntRegister),

    Jmp(String),
    JCond(Condition, String),
    SetCond(Condition, IntRegister),

    //////////////////////////
    //////// Integers ////////
    //////////////////////////
    Mov(IntValue, IntValue),

    Add(IntValue, IntValue),
    Sub(IntValue, IntValue),
    Imul(IntValue, IntValue),
    Idiv(IntValue),

    Inc(IntValue),
    Dec(IntValue),
    Neg(IntValue),

    /// Logical shift left
    Shl(IntValue, IntValue),
    /// Arithmetic shift right
    Sar(IntValue, IntValue),

    And(IntValue, IntValue),
    Or(IntValue, IntValue),
    Xor(IntValue, IntValue),

    Cmp(IntValue, IntValue),
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
            Instruction::Global(name) => writeln!(f, "{:11} {name}", ".global"),
            Instruction::Section(section) => writeln!(f, "\n{:11} {section}", ".section"),
            Instruction::Symbol(name) => writeln!(f, "\n{name}:"),
            Instruction::Byte(num) => writeln!(f, "    {:11} {num:#04x}  # = {num}", ".byte"),
            Instruction::Short(num) => writeln!(f, "    {:11} {num:#06x}  # = {num}", ".short"),
            Instruction::Long(num) => writeln!(f, "    {:11} {num:#010x}  # = {num}", ".long"),
            Instruction::QuadInt(num) => writeln!(f, "    {:11} {num:#018x}  # = {num}", ".quad"),
            Instruction::QuadFloat(num) => {
                writeln!(
                    f,
                    "    {:11} {bits:#018x}  # = {num}{zero}",
                    ".quad",
                    zero = if num.fract() == 0.0 { ".0" } else { "" },
                    bits = num.to_bits()
                )
            }
            Instruction::Octa(num) => writeln!(f, "    {:11} {num:#034x}\t# = {num}", ".octa"),
            Instruction::Leave => writeln!(f, "    leave"),
            Instruction::Ret => writeln!(f, "    ret"),
            Instruction::Syscall => writeln!(f, "    syscall"),
            Instruction::Call(symbol) => writeln!(f, "    {:11} {symbol}", "call"),
            Instruction::Push(reg) => writeln!(f, "    {:11} {reg}", "push"),
            Instruction::Pop(reg) => writeln!(f, "    {:11} {reg}", "pop "),
            Instruction::Jmp(symbol) => writeln!(f, "    {:11} {symbol}", "jmp "),
            Instruction::JCond(cond, symbol) => writeln!(f, "    j{cond:10} {symbol}"),
            Instruction::SetCond(cond, reg) => writeln!(f, "    set{cond:8} {reg}"),
            Instruction::Mov(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "mov"),
            Instruction::Add(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "add"),
            Instruction::Sub(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "sub"),
            Instruction::Imul(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "imul"),
            Instruction::Idiv(divisor) => writeln!(f, "    {:11} {divisor}", "idiv"),
            Instruction::Inc(reg) => writeln!(f, "    {:11} {reg}", "inc"),
            Instruction::Dec(reg) => writeln!(f, "    {:11} {reg}", "dec"),
            Instruction::Neg(reg) => writeln!(f, "    {:11} {reg}", "neg"),
            Instruction::Shl(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "shl"),
            Instruction::Sar(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "sar"),
            Instruction::And(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "and"),
            Instruction::Or(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "or"),
            Instruction::Xor(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "xor"),
            Instruction::Cmp(left, right) => writeln!(f, "    {:11} {left}, {right}", "cmp"),
            Instruction::Cvtsi2sd(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "cvtsi2sd"),
            Instruction::Movsd(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "movsd"),
            Instruction::Addsd(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "addsd"),
            Instruction::Subsd(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "subsd"),
            Instruction::Mulsd(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "mulsd"),
            Instruction::Divsd(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "divsd"),
            Instruction::Xorpd(dest, src) => writeln!(f, "    {:11} {dest}, {src}", "xorpd"),
            Instruction::Ucomisd(left, right) => {
                writeln!(f, "    {:11} {left}, {right}", "ucomisd")
            }
            Instruction::Cvttsd2si(dest, src) => {
                writeln!(f, "    {:11} {dest}, {src}", "cvttsd2si")
            }
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
    use crate::{
        register::FloatRegister,
        value::{Offset, Pointer, Size},
    };

    use super::*;

    #[test]
    fn test() {
        let instrs = [
            Instruction::Cmp(IntValue::Register(IntRegister::Rsi), IntValue::Immediate(0)),
            Instruction::JCond(Condition::Less, "return_0".to_string()),
            Instruction::Mov(IntValue::Register(IntRegister::Rax), IntValue::Immediate(1)),
            Instruction::Ucomisd(
                FloatValue::Register(FloatRegister::Xmm0),
                FloatValue::Ptr(Pointer::new(
                    Size::Qword,
                    IntRegister::Rip,
                    Offset::Symbol("float_0".to_string()),
                )),
            ),
            Instruction::Section(Section::ReadOnlyData),
            Instruction::Symbol("float_127".to_string()),
            Instruction::QuadInt(127_f64.to_bits()),
        ];
        for instr in instrs {
            print!("{instr}");
        }
    }
}

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
            Instruction::Global(name) => writeln!(f, ".global {name}"),
            Instruction::Section(section) => writeln!(f, "\n.section {section}"),
            Instruction::Symbol(name) => writeln!(f, "\n{name}:"),
            Instruction::Byte(num) => writeln!(f, "    .byte\t{num:#04x}\t# = {num}"),
            Instruction::Short(num) => writeln!(f, "    .short\t{num:#06x}\t# = {num}"),
            Instruction::Long(num) => writeln!(f, "    .long\t{num:#010x}\t# = {num}"),
            Instruction::QuadInt(num) => writeln!(f, "    .quad\t{num:#018x}\t# = {num}"),
            Instruction::QuadFloat(num) => {
                writeln!(
                    f,
                    "    .quad\t{bits:#018x}\t# = {num}{zero}",
                    zero = if num.fract() == 0.0 { ".0" } else { "" },
                    bits = num.to_bits()
                )
            }
            Instruction::Octa(num) => writeln!(f, "    .octa\t{num:#034x}\t# = {num}"),
            Instruction::Leave => writeln!(f, "    leave"),
            Instruction::Ret => writeln!(f, "    ret"),
            Instruction::Syscall => writeln!(f, "    syscall"),
            Instruction::Call(symbol) => writeln!(f, "    call\t{symbol}"),
            Instruction::Push(reg) => writeln!(f, "    push\t{reg}"),
            Instruction::Pop(reg) => writeln!(f, "    pop \t{reg}"),
            Instruction::Jmp(symbol) => writeln!(f, "    jmp \t{symbol}"),
            Instruction::JCond(cond, symbol) => writeln!(f, "    j{cond:3}\t{symbol}"),
            Instruction::SetCond(cond, reg) => writeln!(f, "    set{cond}\t{reg}"),
            Instruction::Mov(dest, src) => writeln!(f, "    mov \t{dest}, {src}"),
            Instruction::Add(dest, src) => writeln!(f, "    add \t{dest}, {src}"),
            Instruction::Sub(dest, src) => writeln!(f, "    sub \t{dest}, {src}"),
            Instruction::Imul(dest, src) => writeln!(f, "    imul\t{dest}, {src}"),
            Instruction::Idiv(divisor) => writeln!(f, "    idiv\t{divisor}"),
            Instruction::Inc(reg) => writeln!(f, "    inc \t{reg}"),
            Instruction::Dec(reg) => writeln!(f, "    dec \t{reg}"),
            Instruction::Neg(reg) => writeln!(f, "    neg \t{reg}"),
            Instruction::Shl(dest, src) => writeln!(f, "    shl \t{dest}, {src}"),
            Instruction::Sar(dest, src) => writeln!(f, "    sar \t{dest}, {src}"),
            Instruction::And(dest, src) => writeln!(f, "    and \t{dest}, {src}"),
            Instruction::Or(dest, src) => writeln!(f, "    or  \t{dest}, {src}"),
            Instruction::Xor(dest, src) => writeln!(f, "    xor \t{dest}, {src}"),
            Instruction::Cmp(left, right) => writeln!(f, "    cmp \t{left}, {right}"),
            Instruction::Cvtsi2sd(dest, src) => writeln!(f, "    cvtsi2sd\t{dest}, {src}"),
            Instruction::Movsd(dest, src) => writeln!(f, "    movsd\t{dest}, {src}"),
            Instruction::Addsd(dest, src) => writeln!(f, "    addsd\t{dest}, {src}"),
            Instruction::Subsd(dest, src) => writeln!(f, "    subsd\t{dest}, {src}"),
            Instruction::Mulsd(dest, src) => writeln!(f, "    mulsd\t{dest}, {src}"),
            Instruction::Divsd(dest, src) => writeln!(f, "    divsd\t{dest}, {src}"),
            Instruction::Xorpd(dest, src) => writeln!(f, "    xorpd\t{dest}, {src}"),
            Instruction::Ucomisd(left, right) => writeln!(f, "    ucomisd\t{left}, {right}"),
            Instruction::Cvttsd2si(dest, src) => writeln!(f, "    cvttsd2si\t{dest}, {src}"),
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

use std::{
    borrow::Cow,
    fmt::{self, Display, Formatter},
    rc::Rc,
};

use crate::{
    condition::Condition,
    register::{FloatRegister, IntRegister},
    value::{FloatValue, IntValue, Pointer},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    /// Wraps an instruction with an additional comment
    Commented(Box<Instruction>, Cow<'static, str>),

    IntelSyntax,
    Global(Rc<str>),
    Section(Section),
    Label(Rc<str>, bool),
    Byte(u8),
    Short(u16),
    Long(u32),
    QuadInt(u64),
    QuadFloat(u64),
    Octa(u128),

    /// Load effective address
    Lea(IntRegister, Pointer),

    Leave,
    Ret,
    Call(Rc<str>),
    Push(IntRegister),

    Jmp(Rc<str>),
    JCond(Condition, Rc<str>),
    SetCond(Condition, IntRegister),

    //////////////////////////
    //////// Integers ////////
    //////////////////////////
    Mov(IntValue, IntValue),

    Add(IntValue, IntValue),
    Sub(IntValue, IntValue),
    Imul(IntValue, IntValue),
    /// Divide the 128 bit integer in `%rdx:%rax` by the given value, store the result in `%rax`
    /// and the remainder in `%rdx`
    Idiv(IntValue),

    Inc(IntValue),
    Dec(IntValue),
    /// Negate a signed integer (two's complement)
    Neg(IntValue),
    /// Flip all bits (one's complement)
    Not(IntValue),

    /// Logical shift left
    Shl(IntValue, IntValue),
    /// Arithmetic shift right
    Sar(IntValue, IntValue),

    /// Bitwise AND
    And(IntValue, IntValue),
    /// Bitwise OR
    Or(IntValue, IntValue),
    /// Bitwise XOR
    Xor(IntValue, IntValue),

    /// Compare two integers with subtraction
    Cmp(IntValue, IntValue),
    /// Compare two integers with a bitwise AND
    Test(IntValue, i64),
    /// Convert int to float. (**c**on**v**er**t** **s**igned **i**nteger to **s**calar **d**ouble)
    Cvtsi2sd(FloatValue, IntValue),
    /// Sign-extend 64-bit int in `%rax` to 128-bit int in `%rdx:%rax` (**c**onvert **q**uad to **o**cta)
    Cqo,

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
    Ucomisd(FloatRegister, FloatValue),
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

/////////////////////////////////////////////////

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = f.width().unwrap_or(65);
        match self {
            Instruction::Commented(instr, _) if f.sign_minus() => {
                write!(f, "{instr}")
            }
            Instruction::Commented(instr, comment) => {
                write!(f, "{instr:width$}# {comment}", instr = format!("{instr:#}"))
            }
            Instruction::IntelSyntax => write!(f, ".intel_syntax"),
            Instruction::Global(name) => write!(f, ".global {name}"),
            Instruction::Section(section) => write!(f, "\n.section {section}"),
            Instruction::Label(name, true) => write!(f, "\n{name}:"),
            Instruction::Label(name, false) => write!(f, "{name}:"),
            Instruction::Byte(num) => write!(f, "    {:11} {num:#04x}  # = {num}", ".byte"),
            Instruction::Short(num) => write!(f, "    {:11} {num:#06x}  # = {num}", ".short"),
            Instruction::Long(num) => write!(f, "    {:11} {num:#010x}  # = {num}", ".long"),
            Instruction::QuadInt(num) => write!(f, "    {:11} {num:#018x}  # = {num}", ".quad"),
            Instruction::QuadFloat(num) => {
                let float = f64::from_bits(*num);
                write!(
                    f,
                    "    {:11} {num:#018x}  # = {float}{zero}",
                    ".quad",
                    zero = if float.fract() == 0.0 { ".0" } else { "" },
                )
            }
            Instruction::Octa(num) => write!(f, "    {:11} {num:#034x}\t# = {num}", ".octa"),
            Instruction::Lea(dest, ptr) => write!(f, "    {:11} {dest}, {ptr}", "lea"),
            Instruction::Leave => write!(f, "    leave"),
            Instruction::Ret => write!(f, "    ret"),
            Instruction::Call(label) => write!(f, "    {:11} {label}", "call"),
            Instruction::Push(reg) => write!(f, "    {:11} {reg}", "push"),
            Instruction::Jmp(label) => write!(f, "    {:11} {label}", "jmp "),
            Instruction::JCond(cond, label) => write!(f, "    j{:10} {label}", cond.to_string()),
            Instruction::SetCond(cond, reg) => write!(f, "    set{:8} {reg}", cond.to_string()),
            Instruction::Mov(dest, src) if dest == src => return Ok(()),
            Instruction::Mov(dest, src) => write!(f, "    {:11} {dest}, {src}", "mov"),
            Instruction::Add(dest, src) => write!(f, "    {:11} {dest}, {src}", "add"),
            Instruction::Sub(dest, src) => write!(f, "    {:11} {dest}, {src}", "sub"),
            Instruction::Imul(dest, src) => write!(f, "    {:11} {dest}, {src}", "imul"),
            Instruction::Idiv(divisor) => write!(f, "    {:11} {divisor}", "idiv"),
            Instruction::Inc(reg) => write!(f, "    {:11} {reg}", "inc"),
            Instruction::Dec(reg) => write!(f, "    {:11} {reg}", "dec"),
            Instruction::Neg(reg) => write!(f, "    {:11} {reg}", "neg"),
            Instruction::Not(reg) => write!(f, "    {:11} {reg}", "not"),
            Instruction::Shl(dest, src) => write!(f, "    {:11} {dest}, {src}", "shl"),
            Instruction::Sar(dest, src) => write!(f, "    {:11} {dest}, {src}", "sar"),
            Instruction::And(dest, src) => write!(f, "    {:11} {dest}, {src}", "and"),
            Instruction::Or(dest, src) => write!(f, "    {:11} {dest}, {src}", "or"),
            Instruction::Xor(dest, src) => write!(f, "    {:11} {dest}, {src}", "xor"),
            Instruction::Cmp(left, right) => write!(f, "    {:11} {left}, {right}", "cmp"),
            Instruction::Test(left, right) => write!(f, "    {:11} {left}, {right}", "test"),
            Instruction::Cvtsi2sd(dest, src) => write!(f, "    {:11} {dest}, {src}", "cvtsi2sd"),
            Instruction::Cqo => write!(f, "    cqo"),
            Instruction::Movsd(dest, src) if dest == src => return Ok(()),
            Instruction::Movsd(dest, src) => write!(f, "    {:11} {dest}, {src}", "movsd"),
            Instruction::Addsd(dest, src) => write!(f, "    {:11} {dest}, {src}", "addsd"),
            Instruction::Subsd(dest, src) => write!(f, "    {:11} {dest}, {src}", "subsd"),
            Instruction::Mulsd(dest, src) => write!(f, "    {:11} {dest}, {src}", "mulsd"),
            Instruction::Divsd(dest, src) => write!(f, "    {:11} {dest}, {src}", "divsd"),
            Instruction::Xorpd(dest, src) => write!(f, "    {:11} {dest}, {src}", "xorpd"),
            Instruction::Ucomisd(left, right) => {
                write!(f, "    {:11} {left}, {right}", "ucomisd")
            }
            Instruction::Cvttsd2si(dest, src) => {
                write!(f, "    {:11} {dest}, {src}", "cvttsd2si")
            }
        }?;
        // if not alternate, append a newline
        match f.alternate() {
            true => Ok(()),
            false => writeln!(f),
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

/////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use crate::{
        condition::Condition,
        register::FloatRegister,
        value::{Offset, Pointer, Size},
    };

    use super::*;

    #[test]
    fn test() {
        let instrs = [
            Instruction::Cmp(IntValue::Register(IntRegister::Rsi), IntValue::Immediate(0)),
            Instruction::JCond(Condition::Less, "return_0".into()),
            Instruction::Mov(IntValue::Register(IntRegister::Rax), IntValue::Immediate(1)),
            Instruction::Ucomisd(
                FloatRegister::Xmm0,
                FloatValue::Ptr(Pointer::new(
                    Size::Qword,
                    IntRegister::Rip,
                    Offset::Label("float_0".into()),
                )),
            ),
            Instruction::Section(Section::ReadOnlyData),
            Instruction::Label("float_127".into(), true),
            Instruction::QuadInt(127_f64.to_bits()),
        ];
        for instr in instrs {
            print!("{instr}");
        }
    }
}

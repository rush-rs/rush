use std::{borrow::Cow, fmt::Display, rc::Rc};

use rush_analyzer::InfixOp;

use crate::register::{FloatRegister, IntRegister};

pub enum CommentConfig {
    NoComments,
    Emit { line_width: u8 },
}

pub(crate) struct Block<'tree> {
    pub(crate) label: Rc<str>,
    /// Holds the block's instructions.
    /// The first element of the tuple is the instruction, the second element is a optional comment.
    pub(crate) instructions: Vec<(Instruction, Option<Cow<'tree, str>>)>,
    /// Specifies whether the current block is terminated
    pub(crate) is_terminated: bool,
}

impl<'tree> Block<'tree> {
    pub(crate) fn display(&self, config: &CommentConfig) -> String {
        format!(
            "\n{}:\n{}",
            self.label,
            self.instructions
                .iter()
                .map(|(i, comment)| (i.to_string(), comment))
                .filter(|(i, _)| !i.is_empty())
                .map(|(i, comment)| {
                    match (comment.to_owned(), config) {
                        (Some(msg), CommentConfig::Emit { line_width }) => format!(
                            "    {:width$} # {msg}\n",
                            i.replace('\n', "\n    "),
                            width = *line_width as usize
                        ),
                        (None, _) | (_, CommentConfig::NoComments) => {
                            format!("    {}\n", i.replace('\n', "\n    "))
                        }
                    }
                })
                .collect::<String>()
        )
    }
}

impl<'tree> Block<'tree> {
    pub(crate) fn new(label: Rc<str>) -> Self {
        Self {
            label,
            instructions: vec![],
            is_terminated: false,
        }
    }
}

pub enum Instruction {
    BranchRegister(IntRegister),
    Brasl(IntRegister, Cow<'static, str>),
    Comment(Cow<'static, str>),

    Jmp(Rc<str>),
    // BrCond(Condition, IntRegister, IntRegister, Rc<str>),
    CompareIReg(IntRegister, IntRegister),
    CompareIntImm(IntRegister, i8),
    BranchEq(Rc<str>),

    // base integer instructions
    // SetIntCondition(Condition, IntRegister, IntRegister, IntRegister),
    // Snez(IntRegister, IntRegister),
    // Seqz(IntRegister, IntRegister),

    Lghi(IntRegister, i64),
    La(IntRegister, Rc<str>),
    Lgr(IntRegister, IntRegister),
    // Neg(IntRegister, IntRegister),
    // Not(IntRegister, IntRegister),
    Add64(IntRegister, IntRegister),
    Sub64(IntRegister, IntRegister),
    Mul64(IntRegister, IntRegister),
    Div64(IntRegister, IntRegister),
    Ahi(IntRegister, i8),
    Aghi(IntRegister, i8),
    // Sub(IntRegister, IntRegister, IntRegister),
    // Mul(IntRegister, IntRegister, IntRegister),
    // Div(IntRegister, IntRegister, IntRegister),
    // Rem(IntRegister, IntRegister, IntRegister),
    // Xor(IntRegister, IntRegister, IntRegister),
    // Or(IntRegister, IntRegister, IntRegister),
    // And(IntRegister, IntRegister, IntRegister),
    // Sll(IntRegister, IntRegister, IntRegister),
    // Slli(IntRegister, IntRegister, i64),
    // Sra(IntRegister, IntRegister, IntRegister),
    // Srai(IntRegister, IntRegister, i64),

    // load / store operations
    // Lb(IntRegister, Pointer),
    Load8(IntRegister, Pointer),
    Load64(IntRegister, Pointer),
    Store8(IntRegister, Pointer),
    Store64(IntRegister, Pointer),

    // floats (arithmetic instructions use `.d` suffix)
    // SetFloatCondition(Condition, IntRegister, FloatRegister, FloatRegister),
    // FNeg(FloatRegister, FloatRegister),
    // Fld(FloatRegister, Pointer),
    // Fsd(FloatRegister, Pointer),
    // Fmv(FloatRegister, FloatRegister),
    // Fadd(FloatRegister, FloatRegister, FloatRegister),
    // Fsub(FloatRegister, FloatRegister, FloatRegister),
    // Fmul(FloatRegister, FloatRegister, FloatRegister),
    // Fdiv(FloatRegister, FloatRegister, FloatRegister),

    // casts
    // CastIntToFloat(FloatRegister, IntRegister),
    // CastFloatToInt(IntRegister, FloatRegister),
    // CastByteToFloat(FloatRegister, IntRegister),
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::BranchRegister(reg) => write!(f, "br {reg}"),
            Instruction::Brasl(reg, callee) => write!(f, "brasl {reg}, {callee}"),
            Instruction::Comment(msg) => write!(f, "# {msg}"),
            Instruction::Jmp(label) => write!(f, "j {label}"),
            // Instruction::BrCond(cond, l, r, lbl) => write!(f, "b{cond} {l}, {r}, {lbl}"),
            Self::CompareIReg(a, b) => write!(f, "cgr {a}, {b}"),
            Self::CompareIntImm(x, v) => write!(f, "chi {x}, {v}"),
            Instruction::BranchEq(lbl) => write!(f, "je, {lbl}"),
            Instruction::Lghi(dest, val) => write!(f, "lghi {dest}, {val}"),
            Instruction::Lgr(dest, src) => write!(f, "lgr {dest}, {src}"),
            Instruction::Ahi(dest, v) => write!(f, "ahi {dest}, {v}"),
            Instruction::Aghi(dest, v) => write!(f, "aghi {dest}, {v}"),
            // Instruction::SetIntCondition(cond, dest, l, r) => match cond {
            //     Condition::Lt => write!(f, "slt {dest}, {l}, {r}"),
            //     // Because RISC-V does not support the sle instruction, it is emulated here
            //     Condition::Le => {
            //         writeln!(f, "# begin sle")?;
            //         writeln!(f, "xor t6, {l}, {r}")?;
            //         writeln!(f, "seqz t6, t6")?;
            //         writeln!(f, "slt {dest}, {l}, {r}")?;
            //         writeln!(f, "or {dest}, {dest}, t6")?;
            //         write!(f, "# end sle")
            //     }
            //     // Because RISC-V does not support the seq instruction, it is emulated here
            //     Condition::Eq => {
            //         writeln!(f, "# begin seq")?;
            //         writeln!(f, "xor {dest}, {l}, {r}")?;
            //         writeln!(f, "seqz {dest}, {dest}")?;
            //         write!(f, "# end seq")
            //     }
            //     // Because RISC-V does not support the sne instruction, it is emulated here
            //     Condition::Ne => {
            //         writeln!(f, "# begin sne")?;
            //         writeln!(f, "xor {dest}, {l}, {r}")?;
            //         writeln!(f, "snez {dest}, {dest}")?;
            //         write!(f, "# end sne")
            //     }
            //     Condition::Gt => write!(f, "sgt {dest}, {l}, {r}"),
            //     Condition::Ge => {
            //         writeln!(f, "# begin sge")?;
            //         writeln!(f, "xor t6, {l}, {r}")?;
            //         writeln!(f, "seqz t6, t6")?;
            //         writeln!(f, "sgt {dest}, {l}, {r}")?;
            //         writeln!(f, "or {dest}, {dest}, t6")?;
            //         write!(f, "# end sge")
            //     }
            // },
            // TODO: check that this works!.
            Instruction::La(dest, label) => write!(f, "la {dest}, {label}"),
            Instruction::Add64(dest_lhs, rhs) => write!(f, "agr {dest_lhs}, {rhs}"),
            Instruction::Sub64(dest_lhs, rhs) => write!(f, "sgr {dest_lhs}, {rhs}"),
            Instruction::Mul64(dest_lhs, rhs) => write!(f, "msgr {dest_lhs}, {rhs}"),
            Instruction::Div64(dest_lhs, rhs) => write!(f, "dsgr {dest_lhs}, {rhs}"),
            // Instruction::Addi(dest, src, imm) => write!(f, "addi {dest}, {src}, {imm}"),
            // Instruction::Sub(dest, lhs, rhs) => write!(f, "sub {dest}, {lhs}, {rhs}"),
            // Instruction::Mul(dest, lhs, rhs) => write!(f, "mul {dest}, {lhs}, {rhs}"),
            // Instruction::Div(dest, lhs, rhs) => write!(f, "div {dest}, {lhs}, {rhs}"),
            // Instruction::Rem(dest, lhs, rhs) => write!(f, "rem {dest}, {lhs}, {rhs}"),
            // Instruction::Xor(dest, lhs, rhs) => write!(f, "xor {dest}, {lhs}, {rhs}"),
            // Instruction::Or(dest, lhs, rhs) => write!(f, "or {dest}, {lhs}, {rhs}"),
            // Instruction::And(dest, lhs, rhs) => write!(f, "and {dest}, {lhs}, {rhs}"),
            // Instruction::Sll(dest, lhs, rhs) => write!(f, "sll {dest}, {lhs}, {rhs}"),
            // Instruction::Slli(dest, lhs, amount) => write!(f, "slli {dest}, {lhs}, {amount}"),
            // Instruction::Sra(dest, lhs, rhs) => write!(f, "sra {dest}, {lhs},{rhs}"),
            // Instruction::Srai(dest, lhs, amount) => write!(f, "srai {dest}, {lhs}, {amount}"),
            // Instruction::Lb(dest, ptr) => write!(f, "lb {dest}, {ptr}"),
            Instruction::Load8(dest, ptr) => write!(f, "lb {dest}, {ptr}"),
            Instruction::Load64(dest, ptr) => write!(f, "lg {dest}, {ptr}"),
            // TODO: check that this is not broken.
            // meaning: register vs. label.
            Instruction::Store8(src, ptr) => match ptr {
                Pointer::Register(_, _) => write!(f, "stc {src}, {ptr}"),
                Pointer::Label(_) => write!(f, "stc {src}, {ptr}, t6"),
            },
            Instruction::Store64(src, ptr) => match ptr {
                Pointer::Register(_, _) => write!(f, "stg {src}, {ptr}"),
                Pointer::Label(_) => write!(f, "stg {src}, {ptr}, t6"),
            },
            // Instruction::Fld(dest, ptr) => match ptr {
            //     Pointer::Register(_, _) => write!(f, "fld {dest}, {ptr}"),
            //     Pointer::Label(_) => write!(f, "fld {dest}, {ptr}, t6"),
            // },
            // Instruction::Fsd(src, ptr) => match ptr {
            //     Pointer::Register(_, _) => write!(f, "fsd {src}, {ptr}"),
            //     Pointer::Label(_) => write!(f, "fsd {src}, {ptr}, t6"),
            // },
            // Instruction::Fadd(dest, lhs, rhs) => write!(f, "fadd.d {dest}, {lhs}, {rhs}"),
            // Instruction::Fsub(dest, lhs, rhs) => write!(f, "fsub.d {dest}, {lhs}, {rhs}"),
            // Instruction::Fmul(dest, lhs, rhs) => write!(f, "fmul.d {dest}, {lhs}, {rhs}"),
            // Instruction::Fdiv(dest, lhs, rhs) => write!(f, "fdiv.d {dest}, {lhs}, {rhs}"),
            // Instruction::SetFloatCondition(cond, dest, l, r) => match cond {
            //     Condition::Lt => write!(f, "flt.d {dest}, {l}, {r}"),
            //     Condition::Le => write!(f, "fle.d {dest}, {l}, {r}"),
            //     Condition::Gt => write!(f, "fgt.d {dest}, {l}, {r}"),
            //     Condition::Ge => write!(f, "fge.d {dest}, {l}, {r}"),
            //     Condition::Eq => write!(f, "feq.d {dest}, {l}, {r}"),
            //     Condition::Ne => {
            //         writeln!(f, "feq.d {dest}, {l}, {r}")?;
            //         write!(f, "seqz {dest}, {dest}")
            //     }
            // },
            // Instruction::Snez(dest, arg) => write!(f, "snez {dest}, {arg}"),
            // Instruction::Seqz(dest, arg) => write!(f, "seqz {dest}, {arg}"),
            // Instruction::Lgr(dest, src) => {
            //     if dest != src {
            //         write!(f, "mv {dest}, {src}")
            //     } else {
            //         write!(f, "")
            //     }
            // }
            // Instruction::Fmv(dest, src) => {
            //     if dest != src {
            //         write!(f, "fmv.d {dest}, {src}")
            //     } else {
            //         write!(f, "")
            //     }
            // }
            // Instruction::CastIntToFloat(dest, src) => write!(f, "fcvt.d.l {dest}, {src}"),
            // Instruction::CastByteToFloat(dest, src) => write!(f, "fcvt.d.wu {dest}, {src}"),
            // // rounding mode: round towards zero
            // Instruction::CastFloatToInt(dest, src) => write!(f, "fcvt.l.d {dest}, {src}, rtz"),
            // Instruction::Neg(dest, src) => write!(f, "neg {dest}, {src}"),
            // Instruction::Not(dest, src) => write!(f, "not {dest}, {src}"),
            // Instruction::FNeg(dest, src) => write!(f, "fneg.d {dest}, {src}"),
        }
    }
}

// pub enum Condition {
//     Lt,
//     Le,
//     Eq,
//     Ne,
//     Gt,
//     Ge,
// }
//
// impl Display for Condition {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(
//             f,
//             "{}",
//             match self {
//                 Self::Lt => "lt",
//                 Self::Le => "le",
//                 Self::Gt => "gt",
//                 Self::Ge => "ge",
//                 Self::Eq => "eq",
//                 Self::Ne => "ne",
//             }
//         )
//     }
// }
//
// impl From<InfixOp> for Condition {
//     fn from(src: InfixOp) -> Self {
//         match src {
//             InfixOp::Eq => Self::Eq,
//             InfixOp::Neq => Self::Ne,
//             InfixOp::Lt => Self::Lt,
//             InfixOp::Lte => Self::Le,
//             InfixOp::Gt => Self::Gt,
//             InfixOp::Gte => Self::Ge,
//             _ => unreachable!("infixOp {src} is not used in this way"),
//         }
//     }
// }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pointer {
    Register(IntRegister, i64),
    Label(Rc<str>),
}

impl Display for Pointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Register(reg, offset) => write!(f, "{}({})", offset, reg),
            Self::Label(label) => write!(f, "{label}"),
        }
    }
}

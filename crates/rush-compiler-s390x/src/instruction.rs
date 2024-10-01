use std::{borrow::Cow, fmt::Display, rc::Rc};

use crate::register::{FloatRegister, IntRegister, Register};

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

#[derive(Clone)]
pub enum Instruction {
    Comment(Cow<'static, str>),
    BranchRegister(IntRegister),
    Brasl(IntRegister, Cow<'static, str>),
    Jmp(Rc<str>),
    Cdbr(Register, Register),
    Compare(Register, Register),
    CompareIntImm(IntRegister, i8),
    BranchEq(Rc<str>),
    BranchNotEq(Rc<str>),
    BranchLessThan(Rc<str>),
    BranchGreaterThan(Rc<str>),
    BranchNotGreaterThan(Rc<str>),
    BranchNotGreaterEq(Rc<str>),
    BranchNotLessThan(Rc<str>),
    BranchNotLessEq(Rc<str>),
    Lghi(IntRegister, i16),
    Lgfi(IntRegister, i64),
    Lgr(Register, Register),
    Lder(FloatRegister, FloatRegister),
    Oilf(IntRegister, u32),  // Or immediate (low)
    Llihf(IntRegister, u32), // Load logical high
    Add64(IntRegister, IntRegister),
    Sub64(IntRegister, IntRegister),
    Mul64(IntRegister, IntRegister),
    Div64(IntRegister, IntRegister),
    Adbr(FloatRegister, FloatRegister),
    Sdbr(FloatRegister, FloatRegister),
    Mdbr(FloatRegister, FloatRegister),
    Ddbr(FloatRegister, FloatRegister),
    Xgr(IntRegister, IntRegister),
    Ngr(IntRegister, IntRegister),
    Ogr(IntRegister, IntRegister),
    Lcr(IntRegister, IntRegister),
    Xilf(IntRegister, i8),
    Nilf(IntRegister, i8),
    Aghi(IntRegister, i16),
    ShiftRightSingleLogicalImm(IntRegister, i8),
    ShiftRightSingleLogical(IntRegister, IntRegister, i8, IntRegister),
    ShiftLeftSingleLogical(IntRegister, IntRegister, i8, IntRegister),
    Load8(IntRegister, IntRegisterPointer),
    Load64(IntRegister, IntRegisterPointer),
    Load(Register, IntRegisterPointer),
    LoadAddrRelativeLong(IntRegister, Rc<str>),
    Std(Register, IntRegisterPointer),
    Store8(IntRegister, IntRegisterPointer),
    Store64(IntRegister, IntRegisterPointer),
    ConvertToFixed(IntRegister, u8, FloatRegister),
    ConvertFromFixed(FloatRegister, IntRegister),
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::BranchRegister(reg) => write!(f, "br {reg}"),
            Instruction::Brasl(reg, callee) => write!(f, "brasl {reg}, {callee}"),
            Instruction::Comment(msg) => write!(f, "# {msg}"),
            Instruction::Jmp(label) => write!(f, "j {label}"),
            Self::Cdbr(a, b) => write!(f, "cdbr {a}, {b}"),
            Self::Compare(a, b) => write!(f, "cgr {a}, {b}"),
            Self::CompareIntImm(x, v) => write!(f, "chi {x}, {v}"),
            Instruction::BranchNotEq(lbl) => write!(f, "jne {lbl}"),
            Instruction::BranchEq(lbl) => write!(f, "je {lbl}"),
            Instruction::BranchLessThan(lbl) => write!(f, "jl {lbl}"),
            Instruction::BranchGreaterThan(lbl) => write!(f, "jh {lbl}"),
            Instruction::BranchNotGreaterThan(lbl) => write!(f, "jnh {lbl}"),
            Instruction::BranchNotLessThan(lbl) => write!(f, "jnl {lbl}"),
            Instruction::BranchNotLessEq(lbl) => write!(f, "jnle {lbl}"),
            Instruction::BranchNotGreaterEq(lbl) => write!(f, "jnhe {lbl}"),
            Instruction::Lghi(dest, val) => write!(f, "lghi {dest}, {val}"),
            Instruction::Lgfi(dest, val) => write!(f, "lgfi {dest}, {val}"),
            Instruction::Lgr(dest, src) => write!(f, "lgr {dest}, {src}"),
            Instruction::Lder(dest, src) => write!(f, "lder {dest}, {src}"),
            Instruction::Oilf(dest, v) => write!(f, "oilf {dest}, {v}"),
            Instruction::Llihf(dest, v) => write!(f, "llihf {dest}, {v}"),
            Instruction::Aghi(dest, v) => write!(f, "aghi {dest}, {v}"),
            Instruction::ShiftRightSingleLogicalImm(reg, disp) => write!(f, "srl {reg}, {disp}"),
            Instruction::ShiftRightSingleLogical(dest, source, disp, bas) => {
                write!(f, "srlg {dest}, {source}, {disp}({bas})")
            }
            Instruction::ShiftLeftSingleLogical(dest, source, disp, bas) => {
                write!(f, "sllg {dest}, {source}, {disp}({bas})")
            }
            Instruction::Add64(dest_lhs, rhs) => write!(f, "agr {dest_lhs}, {rhs}"),
            Instruction::Sub64(dest_lhs, rhs) => write!(f, "sgr {dest_lhs}, {rhs}"),
            Instruction::Mul64(dest_lhs, rhs) => write!(f, "msgr {dest_lhs}, {rhs}"),
            Instruction::Div64(dest_lhs, rhs) => write!(f, "dsgr {dest_lhs}, {rhs}"),
            Instruction::Adbr(dest_lhs, rhs) => write!(f, "adbr {dest_lhs}, {rhs}"),
            Instruction::Sdbr(dest_lhs, rhs) => write!(f, "sdbr {dest_lhs}, {rhs}"),
            Instruction::Mdbr(dest_lhs, rhs) => write!(f, "mdbr {dest_lhs}, {rhs}"),
            Instruction::Ddbr(dest_lhs, rhs) => write!(f, "ddbr {dest_lhs}, {rhs}"),
            Instruction::Xgr(dest_lhs, rhs) => write!(f, "xgr {dest_lhs}, {rhs}"),
            Instruction::Ngr(dest_lhs, rhs) => write!(f, "ngr {dest_lhs}, {rhs}"),
            Instruction::Ogr(lhs, rhs) => write!(f, "ogr {lhs}, {rhs}"),
            Instruction::Load8(dest, ptr) => write!(f, "lb {dest}, {ptr}"),
            Instruction::Load64(dest, ptr) => write!(f, "lg {dest}, {ptr}"),
            Instruction::Load(dest, ptr) => write!(f, "ld {dest}, {ptr}"),
            Instruction::LoadAddrRelativeLong(dest, label) => write!(f, "larl {dest}, {label}"),
            Instruction::Std(src, ptr) => write!(f, "std {src}, {ptr}"),
            Instruction::Store8(src, ptr) => write!(f, "stc {src}, {ptr}"),
            Instruction::Store64(src, ptr) => write!(f, "stg {src}, {ptr}"),
            Instruction::ConvertToFixed(dest, rounding, src) => {
                write!(f, "cgdbr {dest}, {rounding}, {src}")
            }
            Instruction::ConvertFromFixed(dest, src) => write!(f, "cdgbr  {dest}, {src}"),
            Instruction::Lcr(dest, src) => write!(f, "lcr {dest}, {src}"),
            Instruction::Xilf(dest, v) => write!(f, "xilf {dest}, {v}"),
            Instruction::Nilf(dest, v) => write!(f, "nilf {dest}, {v}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pointer {
    Register(IntRegisterPointer),
    Label(Rc<str>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IntRegisterPointer(pub IntRegister, pub i64);

impl Display for IntRegisterPointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.1, self.0)
    }
}

impl IntRegisterPointer {
    pub(crate) fn offset(&self) -> i64 {
        self.1
    }

    pub(crate) fn reg(&self) -> IntRegister {
        self.0
    }
}

impl Display for Pointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Register(reg) => write!(f, "{}", reg),
            Self::Label(label) => write!(f, "{label}"),
        }
    }
}

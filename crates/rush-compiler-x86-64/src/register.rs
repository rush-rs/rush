#![allow(dead_code)] // TODO: remove this attribute

use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntRegister {
    Rip,

    Rax,
    Rbx,
    Rcx,
    Rdx,
    Rsi,
    Rdi,
    Rbp,
    Rsp,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,

    Eax,
    Ebx,
    Ecx,
    Edx,
    Esi,
    Edi,
    Ebp,
    Esp,
    R8d,
    R9d,
    R10d,
    R11d,
    R12d,
    R13d,
    R14d,
    R15d,

    Ax,
    Bx,
    Cx,
    Dx,
    Si,
    Di,
    Bp,
    Sp,
    R8w,
    R9w,
    R10w,
    R11w,
    R12w,
    R13w,
    R14w,
    R15w,

    Al,
    Bl,
    Cl,
    Dl,
    Sil,
    Dil,
    Bpl,
    Spl,
    R8b,
    R9b,
    R10b,
    R11b,
    R12b,
    R13b,
    R14b,
    R15b,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatRegister {
    Xmm0,
    Xmm1,
    Xmm2,
    Xmm3,
    Xmm4,
    Xmm5,
    Xmm6,
    Xmm7,
    Xmm8,
    Xmm9,
    Xmm10,
    Xmm11,
    Xmm12,
    Xmm13,
    Xmm14,
    Xmm15,
    Xmm16,
    Xmm17,
    Xmm18,
    Xmm19,
    Xmm20,
    Xmm21,
    Xmm22,
    Xmm23,
    Xmm24,
    Xmm25,
    Xmm26,
    Xmm27,
    Xmm28,
    Xmm29,
    Xmm30,
    Xmm31,
}

impl Display for IntRegister {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", format!("{self:?}").to_lowercase())
    }
}

impl Display for FloatRegister {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", format!("{self:?}").to_lowercase())
    }
}

impl IntRegister {
    pub fn byte_variant(&self) -> Self {
        match self {
            Self::Rip => panic!("no byte variant for `%rip` register"),
            Self::Rax | Self::Eax | Self::Ax | Self::Al => Self::Al,
            Self::Rbx | Self::Ebx | Self::Bx | Self::Bl => Self::Bl,
            Self::Rcx | Self::Ecx | Self::Cx | Self::Cl => Self::Cl,
            Self::Rdx | Self::Edx | Self::Dx | Self::Dl => Self::Dl,
            Self::Rsi | Self::Esi | Self::Si | Self::Sil => Self::Sil,
            Self::Rdi | Self::Edi | Self::Di | Self::Dil => Self::Dil,
            Self::Rbp | Self::Ebp | Self::Bp | Self::Bpl => Self::Bpl,
            Self::Rsp | Self::Esp | Self::Sp | Self::Spl => Self::Spl,
            Self::R8 | Self::R8d | Self::R8w | Self::R8b => Self::R8b,
            Self::R9 | Self::R9d | Self::R9w | Self::R9b => Self::R9b,
            Self::R10 | Self::R10d | Self::R10w | Self::R10b => Self::R10b,
            Self::R11 | Self::R11d | Self::R11w | Self::R11b => Self::R11b,
            Self::R12 | Self::R12d | Self::R12w | Self::R12b => Self::R12b,
            Self::R13 | Self::R13d | Self::R13w | Self::R13b => Self::R13b,
            Self::R14 | Self::R14d | Self::R14w | Self::R14b => Self::R14b,
            Self::R15 | Self::R15d | Self::R15w | Self::R15b => Self::R15b,
        }
    }

    pub fn word_variant(&self) -> Self {
        match self {
            Self::Rip => panic!("no word variant for `%rip` register"),
            Self::Rax | Self::Eax | Self::Ax | Self::Al => Self::Ax,
            Self::Rbx | Self::Ebx | Self::Bx | Self::Bl => Self::Bx,
            Self::Rcx | Self::Ecx | Self::Cx | Self::Cl => Self::Cx,
            Self::Rdx | Self::Edx | Self::Dx | Self::Dl => Self::Dx,
            Self::Rsi | Self::Esi | Self::Si | Self::Sil => Self::Si,
            Self::Rdi | Self::Edi | Self::Di | Self::Dil => Self::Di,
            Self::Rbp | Self::Ebp | Self::Bp | Self::Bpl => Self::Bp,
            Self::Rsp | Self::Esp | Self::Sp | Self::Spl => Self::Sp,
            Self::R8 | Self::R8d | Self::R8w | Self::R8b => Self::R8w,
            Self::R9 | Self::R9d | Self::R9w | Self::R9b => Self::R9w,
            Self::R10 | Self::R10d | Self::R10w | Self::R10b => Self::R10w,
            Self::R11 | Self::R11d | Self::R11w | Self::R11b => Self::R11w,
            Self::R12 | Self::R12d | Self::R12w | Self::R12b => Self::R12w,
            Self::R13 | Self::R13d | Self::R13w | Self::R13b => Self::R13w,
            Self::R14 | Self::R14d | Self::R14w | Self::R14b => Self::R14w,
            Self::R15 | Self::R15d | Self::R15w | Self::R15b => Self::R15w,
        }
    }

    pub fn dword_variant(&self) -> Self {
        match self {
            Self::Rip => panic!("no dword variant for `%rip` register"),
            Self::Rax | Self::Eax | Self::Ax | Self::Al => Self::Eax,
            Self::Rbx | Self::Ebx | Self::Bx | Self::Bl => Self::Ebx,
            Self::Rcx | Self::Ecx | Self::Cx | Self::Cl => Self::Ecx,
            Self::Rdx | Self::Edx | Self::Dx | Self::Dl => Self::Edx,
            Self::Rsi | Self::Esi | Self::Si | Self::Sil => Self::Esi,
            Self::Rdi | Self::Edi | Self::Di | Self::Dil => Self::Edi,
            Self::Rbp | Self::Ebp | Self::Bp | Self::Bpl => Self::Ebp,
            Self::Rsp | Self::Esp | Self::Sp | Self::Spl => Self::Esp,
            Self::R8 | Self::R8d | Self::R8w | Self::R8b => Self::R8d,
            Self::R9 | Self::R9d | Self::R9w | Self::R9b => Self::R9d,
            Self::R10 | Self::R10d | Self::R10w | Self::R10b => Self::R10d,
            Self::R11 | Self::R11d | Self::R11w | Self::R11b => Self::R11d,
            Self::R12 | Self::R12d | Self::R12w | Self::R12b => Self::R12d,
            Self::R13 | Self::R13d | Self::R13w | Self::R13b => Self::R13d,
            Self::R14 | Self::R14d | Self::R14w | Self::R14b => Self::R14d,
            Self::R15 | Self::R15d | Self::R15w | Self::R15b => Self::R15d,
        }
    }

    pub fn qword_variant(&self) -> Self {
        match self {
            Self::Rip => panic!("no qword variant for `%rip` register"),
            Self::Rax | Self::Eax | Self::Ax | Self::Al => Self::Rax,
            Self::Rbx | Self::Ebx | Self::Bx | Self::Bl => Self::Rbx,
            Self::Rcx | Self::Ecx | Self::Cx | Self::Cl => Self::Rcx,
            Self::Rdx | Self::Edx | Self::Dx | Self::Dl => Self::Rdx,
            Self::Rsi | Self::Esi | Self::Si | Self::Sil => Self::Rsi,
            Self::Rdi | Self::Edi | Self::Di | Self::Dil => Self::Rdi,
            Self::Rbp | Self::Ebp | Self::Bp | Self::Bpl => Self::Rbp,
            Self::Rsp | Self::Esp | Self::Sp | Self::Spl => Self::Rsp,
            Self::R8 | Self::R8d | Self::R8w | Self::R8b => Self::R8,
            Self::R9 | Self::R9d | Self::R9w | Self::R9b => Self::R9,
            Self::R10 | Self::R10d | Self::R10w | Self::R10b => Self::R10,
            Self::R11 | Self::R11d | Self::R11w | Self::R11b => Self::R11,
            Self::R12 | Self::R12d | Self::R12w | Self::R12b => Self::R12,
            Self::R13 | Self::R13d | Self::R13w | Self::R13b => Self::R13,
            Self::R14 | Self::R14d | Self::R14w | Self::R14b => Self::R14,
            Self::R15 | Self::R15d | Self::R15w | Self::R15b => Self::R15,
        }
    }

    pub fn next_param(&self) -> Option<Self> {
        match self {
            Self::Rdi => Some(Self::Rsi),
            Self::Rsi => Some(Self::Rsi),
            Self::Rdx => Some(Self::Rdx),
            Self::Rcx => Some(Self::Rcx),
            Self::R8 => Some(Self::R9),
            Self::R9 => None,

            Self::Edi => Some(Self::Esi),
            Self::Esi => Some(Self::Esi),
            Self::Edx => Some(Self::Edx),
            Self::Ecx => Some(Self::Ecx),
            Self::R8d => Some(Self::R9d),
            Self::R9d => None,

            Self::Di => Some(Self::Si),
            Self::Si => Some(Self::Si),
            Self::Dx => Some(Self::Dx),
            Self::Cx => Some(Self::Cx),
            Self::R8w => Some(Self::R9w),
            Self::R9w => None,

            Self::Dil => Some(Self::Sil),
            Self::Sil => Some(Self::Sil),
            Self::Dl => Some(Self::Dl),
            Self::Cl => Some(Self::Cl),
            Self::R8b => Some(Self::R9b),
            Self::R9b => None,

            reg => panic!("not a param register `{reg}`"),
        }
    }
}

impl FloatRegister {
    pub fn next_param(&self) -> Option<Self> {
        match self {
            Self::Xmm0 => Some(Self::Xmm1),
            Self::Xmm1 => Some(Self::Xmm2),
            Self::Xmm2 => Some(Self::Xmm3),
            Self::Xmm3 => Some(Self::Xmm4),
            Self::Xmm4 => Some(Self::Xmm5),
            Self::Xmm5 => Some(Self::Xmm6),
            Self::Xmm6 => Some(Self::Xmm7),
            Self::Xmm7 => None,
            reg => panic!("not a param register `{reg}`"),
        }
    }
}

use std::fmt::Debug;

use crate::Span;

#[derive(Clone, Copy, PartialEq)]
pub struct Token<'src> {
    pub kind: TokenKind<'src>,
    pub span: Span,
}

impl<'src> Token<'src> {
    pub fn new(kind: TokenKind<'src>, span: Span) -> Self {
        Self { kind, span }
    }
}

impl<'src> Debug for Token<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<15} (l:{}:{} -- l:{}:{})",
            format!("{:?}", self.kind),
            self.span.start.line,
            self.span.start.column,
            self.span.end.line,
            self.span.end.column
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenKind<'src> {
    Eof,

    Ident(&'src str),
    Int(i64),
    Float(f64),
    Char(u8),

    True,
    False,
    Fn,
    Let,
    Mut,
    Return,
    If,
    Else,

    LParen,
    RParen,
    LBrace,
    RBrace,

    Arrow,
    Comma,
    Colon,
    Semicolon,

    Not,
    Minus,
    Plus,
    Star,
    Slash,
    Percent,
    Pow,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    Shl,
    Shr,
    BitOr,
    BitAnd,
    BitXor,
    And,
    Or,

    Assign,
    PlusAssign,
    MinusAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    PowAssign,
    ShlAssign,
    ShrAssign,
    BitOrAssign,
    BitAndAssign,
    BitXorAssign,
}

// TODO: implement Display trait on token kind

impl<'src> TokenKind<'src> {
    pub fn spanned(self, span: Span) -> Token<'src> {
        Token { kind: self, span }
    }

    pub(crate) fn prec(&self) -> (u8, u8) {
        match self {
            TokenKind::Assign
            | TokenKind::PlusAssign
            | TokenKind::MinusAssign
            | TokenKind::MulAssign
            | TokenKind::DivAssign
            | TokenKind::RemAssign
            | TokenKind::PowAssign
            | TokenKind::ShlAssign
            | TokenKind::ShrAssign
            | TokenKind::BitOrAssign
            | TokenKind::BitAndAssign
            | TokenKind::BitXorAssign => (1, 2),
            TokenKind::Or => (3, 4),
            TokenKind::And => (5, 6),
            TokenKind::BitOr => (7, 8),
            TokenKind::BitXor => (9, 10),
            TokenKind::BitAnd => (11, 12),
            TokenKind::Eq | TokenKind::Neq => (13, 14),
            TokenKind::Lt | TokenKind::Gt | TokenKind::Lte | TokenKind::Gte => (15, 16),
            TokenKind::Shl | TokenKind::Shr => (17, 18),
            TokenKind::Plus | TokenKind::Minus => (19, 20),
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => (21, 22),
            TokenKind::Pow => (24, 23),
            TokenKind::LParen => (25, 26),
            _ => (0, 0),
        }
    }
}

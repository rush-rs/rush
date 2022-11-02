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
}

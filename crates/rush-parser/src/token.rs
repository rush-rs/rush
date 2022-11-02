use crate::Span;

pub struct Token<'src> {
    pub kind: TokenKind<'src>,
    pub span: Span,
}

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

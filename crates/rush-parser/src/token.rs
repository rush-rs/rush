use std::fmt::{self, Debug, Display, Formatter};

use crate::Span;

#[derive(Clone, Copy, PartialEq, Default)]
pub struct Token<'src> {
    pub kind: TokenKind<'src>,
    pub span: Span,
}

impl<'src> Token<'src> {
    pub fn new(kind: TokenKind<'src>, span: Span) -> Self {
        Self { kind, span }
    }
}

impl Debug for Token<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TokenKind<'src> {
    /// End of file
    #[default]
    Eof,

    /// An identifier
    Ident(&'src str),
    /// An int literal
    Int(i64),
    /// A float literal
    Float(f64),
    /// A char literal
    Char(u8),

    /// `true`
    True,
    /// `false`
    False,
    /// `fn`
    Fn,
    /// `let`
    Let,
    /// `mut`
    Mut,
    /// `return`
    Return,
    /// `loop`
    Loop,
    /// `while`
    While,
    /// `break`
    Break,
    /// `if`
    If,
    /// `else`
    Else,
    /// `as`
    As,

    /// (
    LParen,
    /// )
    RParen,
    /// {
    LBrace,
    /// }
    RBrace,

    /// ->
    Arrow,
    /// ,
    Comma,
    /// :
    Colon,
    /// ;
    Semicolon,

    /// !
    Not,
    /// -
    Minus,
    /// +
    Plus,
    /// *
    Star,
    /// /
    Slash,
    /// %
    Percent,
    /// **
    Pow,
    /// ==
    Eq,
    /// !=
    Neq,
    /// <
    Lt,
    /// >
    Gt,
    /// <=
    Lte,
    /// >=
    Gte,
    /// <<
    Shl,
    /// >>
    Shr,
    /// |
    BitOr,
    /// &
    BitAnd,
    /// ^
    BitXor,
    /// &&
    And,
    /// ||
    Or,

    /// =
    Assign,
    /// +=
    PlusAssign,
    /// -=
    MinusAssign,
    /// *=
    MulAssign,
    /// /=
    DivAssign,
    /// %=
    RemAssign,
    /// **=
    PowAssign,
    /// <<=
    ShlAssign,
    /// >>=
    ShrAssign,
    /// |=
    BitOrAssign,
    /// &=
    BitAndAssign,
    /// ^=
    BitXorAssign,
}

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
            TokenKind::As => (23, 24),
            TokenKind::Pow => (26, 25), // inverse order for right associativity
            TokenKind::LParen => (28, 29),
            _ => (0, 0),
        }
    }
}

impl Display for TokenKind<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eof => write!(f, "EOF"),
            Self::Ident(ident) => write!(f, "{ident}"),
            Self::Int(num) => write!(f, "{num}"),
            Self::Float(num) => write!(f, "{num}"),
            Self::Char(b'\\') => write!(f, "'\\\\'"),
            Self::Char(b'\x08') => write!(f, "'\\b'"),
            Self::Char(b'\n') => write!(f, "'\\n'"),
            Self::Char(b'\r') => write!(f, "'\\r'"),
            Self::Char(b'\t') => write!(f, "'\\t'"),
            Self::Char(b'\'') => write!(f, "'\\''"),
            Self::Char(char @ b' '..=b'~') => write!(f, "'{}'", *char as char),
            Self::Char(char) => write!(f, "'\\x{char:x}'"),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
            Self::Fn => write!(f, "fn"),
            Self::Let => write!(f, "let"),
            Self::Mut => write!(f, "mut"),
            Self::Return => write!(f, "return"),
            Self::Loop => write!(f, "loop"),
            Self::While => write!(f, "while"),
            Self::Break => write!(f, "break"),
            Self::If => write!(f, "if"),
            Self::Else => write!(f, "else"),
            Self::As => write!(f, "as"),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::LBrace => write!(f, "{{"),
            Self::RBrace => write!(f, "}}"),
            Self::Arrow => write!(f, "->"),
            Self::Comma => write!(f, ","),
            Self::Colon => write!(f, ":"),
            Self::Semicolon => write!(f, ";"),
            Self::Not => write!(f, "!"),
            Self::Minus => write!(f, "-"),
            Self::Plus => write!(f, "+"),
            Self::Star => write!(f, "*"),
            Self::Slash => write!(f, "/"),
            Self::Percent => write!(f, "%"),
            Self::Pow => write!(f, "**"),
            Self::Eq => write!(f, "=="),
            Self::Neq => write!(f, "!="),
            Self::Lt => write!(f, "<"),
            Self::Gt => write!(f, ">"),
            Self::Lte => write!(f, "<="),
            Self::Gte => write!(f, ">="),
            Self::Shl => write!(f, "<<"),
            Self::Shr => write!(f, ">>"),
            Self::BitOr => write!(f, "|"),
            Self::BitAnd => write!(f, "&"),
            Self::BitXor => write!(f, "^"),
            Self::And => write!(f, "&&"),
            Self::Or => write!(f, "||"),
            Self::Assign => write!(f, "="),
            Self::PlusAssign => write!(f, "+="),
            Self::MinusAssign => write!(f, "-="),
            Self::MulAssign => write!(f, "*="),
            Self::DivAssign => write!(f, "/="),
            Self::RemAssign => write!(f, "%="),
            Self::PowAssign => write!(f, "**="),
            Self::ShlAssign => write!(f, "<<="),
            Self::ShrAssign => write!(f, ">>="),
            Self::BitOrAssign => write!(f, "|="),
            Self::BitAndAssign => write!(f, "&="),
            Self::BitXorAssign => write!(f, "^="),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_display() {
        assert_eq!(TokenKind::Char(b'\\').to_string(), r"'\\'");
        assert_eq!(TokenKind::Char(0x08).to_string(), r"'\b'");
        assert_eq!(TokenKind::Char(b'\n').to_string(), r"'\n'");
        assert_eq!(TokenKind::Char(b'\r').to_string(), r"'\r'");
        assert_eq!(TokenKind::Char(b'\t').to_string(), r"'\t'");
        assert_eq!(TokenKind::Char(b'\'').to_string(), r"'\''");
        assert_eq!(TokenKind::Char(b'a').to_string(), "'a'");
        assert_eq!(TokenKind::Char(b'0').to_string(), "'0'");
        assert_eq!(TokenKind::Char(b' ').to_string(), "' '");
        assert_eq!(TokenKind::Char(b'~').to_string(), "'~'");
        assert_eq!(TokenKind::Char(0x7f).to_string(), r"'\x7f'");
        assert_eq!(TokenKind::Char(0x1b).to_string(), r"'\x1b'");
    }
}

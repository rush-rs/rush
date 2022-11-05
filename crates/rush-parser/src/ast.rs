use std::fmt::{self, Debug, Display, Formatter};

use crate::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Float,
    Bool,
    Char,
    Unit,
    /// Internal use only, used for diverging expressions
    Never,
    /// Internal use only, used if typecheck could not determine a type
    Unknown,
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Int => "int",
                Self::Float => "float",
                Self::Bool => "bool",
                Self::Char => "char",
                Self::Unit => "()",
                Self::Never => "!",
                Self::Unknown => "{unknown}",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    pub span: Span,
    pub inner: T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program<'src> {
    pub span: Span,
    pub functions: Vec<FunctionDefinition<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDefinition<'src> {
    pub span: Span,
    pub name: Spanned<&'src str>,
    pub params: Vec<(Spanned<&'src str>, Spanned<Type>)>,
    pub return_type: Spanned<Type>,
    pub block: Block<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block<'src> {
    pub span: Span,
    pub stmts: Vec<Statement<'src>>,
    pub expr: Option<Expression<'src>>,
}

impl Block<'_> {
    /// Returns the span responsible for the block's result type
    pub fn result_span(&self) -> Span {
        self.expr.as_ref().map_or_else(
            || self.stmts.last().map_or(self.span, |stmt| stmt.span()),
            |expr| expr.span(),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement<'src> {
    Let(LetStmt<'src>),
    Return(ReturnStmt<'src>),
    Expr(ExprStmt<'src>),
}

impl Statement<'_> {
    pub fn span(&self) -> Span {
        match self {
            Self::Let(stmt) => stmt.span,
            Self::Return(stmt) => stmt.span,
            Self::Expr(stmt) => stmt.span,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetStmt<'src> {
    pub span: Span,
    pub mutable: bool,
    pub name: Spanned<&'src str>,
    pub type_: Option<Spanned<Type>>,
    pub expr: Expression<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReturnStmt<'src> {
    pub span: Span,
    pub expr: Option<Expression<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExprStmt<'src> {
    pub span: Span,
    pub expr: Expression<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression<'src> {
    Block(Box<Block<'src>>),
    If(Box<IfExpr<'src>>),
    Int(Spanned<i64>),
    Float(Spanned<f64>),
    Bool(Spanned<bool>),
    Ident(Spanned<&'src str>),
    Prefix(Box<PrefixExpr<'src>>),
    Infix(Box<InfixExpr<'src>>),
    Assign(Box<AssignExpr<'src>>),
    Call(Box<CallExpr<'src>>),
    Cast(Box<CastExpr<'src>>),
    Grouped(Spanned<Box<Expression<'src>>>),
}

impl Expression<'_> {
    pub fn span(&self) -> Span {
        match self {
            Self::Block(expr) => expr.span,
            Self::If(expr) => expr.span,
            Self::Int(expr) => expr.span,
            Self::Float(expr) => expr.span,
            Self::Bool(expr) => expr.span,
            Self::Ident(expr) => expr.span,
            Self::Prefix(expr) => expr.span,
            Self::Infix(expr) => expr.span,
            Self::Assign(expr) => expr.span,
            Self::Call(expr) => expr.span,
            Self::Cast(expr) => expr.span,
            Self::Grouped(expr) => expr.span,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfExpr<'src> {
    pub span: Span,
    pub cond: Expression<'src>,
    pub then_block: Block<'src>,
    pub else_block: Option<Block<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefixExpr<'src> {
    pub span: Span,
    pub op: PrefixOp,
    pub expr: Expression<'src>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixOp {
    /// !
    Not,
    /// -
    Neg,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InfixExpr<'src> {
    pub span: Span,
    pub lhs: Expression<'src>,
    pub op: InfixOp,
    pub rhs: Expression<'src>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfixOp {
    /// +
    Plus,
    /// -
    Minus,
    /// *
    Mul,
    /// /
    Div,
    /// %
    Rem,
    /// *
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
}

impl Display for InfixOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Plus => "+",
                Self::Minus => "-",
                Self::Mul => "*",
                Self::Div => "/",
                Self::Rem => "%",
                Self::Pow => "**",
                Self::Eq => "==",
                Self::Neq => "!=",
                Self::Lt => "<",
                Self::Gt => ">",
                Self::Lte => "<=",
                Self::Gte => ">=",
                Self::Shl => "<<",
                Self::Shr => ">>",
                Self::BitOr => "|",
                Self::BitAnd => "&",
                Self::BitXor => "^",
                Self::And => "&&",
                Self::Or => "||",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignExpr<'src> {
    pub span: Span,
    pub assignee: Spanned<&'src str>,
    pub op: AssignOp,
    pub expr: Expression<'src>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    /// =
    Basic,
    /// +=
    Plus,
    /// -=
    Minus,
    /// *=
    Mul,
    /// /=
    Div,
    /// %=
    Rem,
    /// **=
    Pow,
    /// <<=
    Shl,
    /// >>=
    Shr,
    /// |=
    BitOr,
    /// &=
    BitAnd,
    /// ^=
    BitXor,
}

impl Display for AssignOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}=",
            match self {
                Self::Basic => "",
                Self::Plus => "+",
                Self::Minus => "-",
                Self::Mul => "*",
                Self::Div => "/",
                Self::Rem => "%",
                Self::Pow => "**",
                Self::Shl => "<<",
                Self::Shr => ">>",
                Self::BitOr => "|",
                Self::BitAnd => "&",
                Self::BitXor => "^",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallExpr<'src> {
    pub span: Span,
    pub func: Spanned<&'src str>,
    pub args: Vec<Expression<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CastExpr<'src> {
    pub span: Span,
    pub expr: Expression<'src>,
    pub type_: Spanned<Type>,
}

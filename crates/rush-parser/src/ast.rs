use std::fmt::{Debug, Display};

use crate::Span;

pub type ParsedProgram<'src> = Program<'src, ParsedStatement<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program<'src, Stmt> {
    pub span: Span,
    pub functions: Vec<FunctionDefinition<'src, Stmt>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

pub type ParsedFunctionDefinition<'src> = FunctionDefinition<'src, ParsedStatement<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDefinition<'src, Stmt> {
    pub span: Span,
    pub name: &'src str,
    pub params: Vec<(&'src str, Type)>,
    pub return_type: Type,
    pub block: Block<Stmt>,
}

pub type ParsedBlock<'src> = Block<ParsedStatement<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block<Stmt> {
    pub span: Span,
    pub stmts: Vec<Stmt>,
}

pub type ParsedStatement<'src> = Statement<'src, ParsedExpression<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement<'src, Expr> {
    Let(LetStmt<'src, Expr>),
    Return(ReturnStmt<Expr>),
    Expr(ExprStmt<Expr>),
}

impl<Expr> Statement<'_, Expr> {
    pub fn span(&self) -> Span {
        match self {
            Self::Let(stmt) => stmt.span,
            Self::Return(stmt) => stmt.span,
            Self::Expr(stmt) => stmt.span,
        }
    }
}

pub type ParsedLetStmt<'src> = LetStmt<'src, ParsedExpression<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LetStmt<'src, Expr> {
    pub span: Span,
    pub mutable: bool,
    pub name: &'src str,
    pub type_: Option<Type>,
    pub expr: Expr,
}

pub type ParsedReturnStmt<'src> = ReturnStmt<ParsedExpression<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReturnStmt<Expr> {
    pub span: Span,
    pub expr: Option<Expr>,
}

pub type ParsedExprStmt<'src> = ExprStmt<ParsedExpression<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprStmt<Expr> {
    pub span: Span,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedExpression<'src>(pub BareParsedExpression<'src>);

impl<'src> From<BareParsedExpression<'src>> for ParsedExpression<'src> {
    fn from(expr: BareParsedExpression<'src>) -> Self {
        Self(expr)
    }
}

pub type BareParsedExpression<'src> =
    Expression<'src, ParsedStatement<'src>, ParsedExpression<'src>>;
#[derive(Debug, Clone, PartialEq)]
pub enum Expression<'src, Stmt, Expr> {
    Block(Box<Block<Stmt>>),
    If(Box<IfExpr<Stmt, Expr>>),
    Int(Atom<i64>),
    Float(Atom<f64>),
    Bool(Atom<bool>),
    Ident(Atom<&'src str>),
    Prefix(Box<PrefixExpr<Expr>>),
    Infix(Box<InfixExpr<Expr>>),
    Assign(Box<AssignExpr<'src, Expr>>),
    Call(Box<CallExpr<Expr>>),
    Cast(Box<CastExpr<Expr>>),
    Grouped(Atom<Box<Expr>>),
}

impl<Stmt, Expr> Expression<'_, Stmt, Expr> {
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

pub type ParsedIfExpr<'src> = IfExpr<ParsedStatement<'src>, ParsedExpression<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfExpr<Stmt, Expr> {
    pub span: Span,
    pub cond: Expr,
    pub then_block: Block<Stmt>,
    pub else_block: Option<Block<Stmt>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atom<T> {
    pub span: Span,
    pub value: T,
}

pub type ParsedPrefixExpr<'src> = PrefixExpr<ParsedExpression<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixExpr<Expr> {
    pub span: Span,
    pub op: PrefixOp,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixOp {
    /// !
    Not,
    /// -
    Neg,
}

pub type ParsedInfixExpr<'src> = InfixExpr<ParsedExpression<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfixExpr<Expr> {
    pub span: Span,
    pub lhs: Expr,
    pub op: InfixOp,
    pub rhs: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

pub type ParsedAssignExpr<'src> = AssignExpr<'src, ParsedExpression<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignExpr<'src, Expr> {
    pub span: Span,
    pub assignee: Atom<&'src str>,
    pub op: AssignOp,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

pub type ParsedCallExpr<'src> = CallExpr<ParsedExpression<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallExpr<Expr> {
    pub span: Span,
    pub expr: Expr,
    pub args: Vec<Expr>,
}

pub type ParsedCastExpr<'src> = CastExpr<ParsedExpression<'src>>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CastExpr<Expr> {
    pub span: Span,
    pub expr: Expr,
    pub type_: Type,
}

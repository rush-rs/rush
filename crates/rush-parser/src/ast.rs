use std::fmt::Debug;

use crate::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Program<'src> {
    pub span: Span,
    pub functions: Vec<FunctionDefinition<'src>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Float,
    Bool,
    Char,
    Unit,
    Never,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDefinition<'src> {
    pub span: Span,
    pub name: &'src str,
    pub params: Vec<(&'src str, Type)>,
    pub return_type: Type,
    pub block: Block<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block<'src> {
    pub span: Span,
    pub stmts: Vec<Statement<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement<'src> {
    Let(LetStmt<'src>),
    Return(ReturnStmt<'src>),
    Expr(ExprStmt<'src>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetStmt<'src> {
    pub span: Span,
    pub mutable: bool,
    pub name: &'src str,
    pub type_: Option<Type>,
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
    Int(AtomExpr<i64>),
    Float(AtomExpr<f64>),
    Bool(AtomExpr<bool>),
    Ident(AtomExpr<&'src str>),
    Prefix(Box<PrefixExpr<'src>>),
    Infix(Box<InfixExpr<'src>>),
    Assign(Box<AssignExpr<'src>>),
    Call(Box<CallExpr<'src>>),
    Cast(Box<CastExpr<'src>>),
    Grouped(Box<Expression<'src>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfExpr<'src> {
    pub span: Span,
    pub cond: Expression<'src>,
    pub then_block: Block<'src>,
    pub else_block: Option<Block<'src>>,
}

#[allow(clippy::derive_partial_eq_without_eq)] // The trait bounds of T don't specify Eq
#[derive(Debug, Clone, PartialEq)]
pub struct AtomExpr<T>
where
    T: Debug + Clone + PartialEq,
{
    pub span: Span,
    pub value: T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefixExpr<'src> {
    pub span: Span,
    pub op: PrefixOp,
    pub expr: Expression<'src>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixOp {
    Not, // !
    Neg, // -
}

#[derive(Debug, Clone, PartialEq)]
pub struct InfixExpr<'src> {
    pub span: Span,
    pub lhs: Expression<'src>,
    pub op: InfixOp,
    pub rhs: Expression<'src>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InfixOp {
    Plus,  // +
    Minus, // -
    Mul,   // *
    Div,   // /
    Rem,   // %
    Pow,   // *

    Eq,  // ==
    Neq, // !=
    Lt,  // <
    Gt,  // >
    Lte, // <=
    Gte, // >=

    Shl,    // <<
    Shr,    // >>
    BitOr,  // |
    BitAnd, // &
    BitXor, // ^

    And, // &&
    Or,  // ||
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignExpr<'src> {
    pub span: Span,
    pub assignee: AtomExpr<&'src str>,
    pub op: AssignOp,
    pub expr: Expression<'src>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignOp {
    Basic,  // =
    Plus,   // +=
    Minus,  // -=
    Mul,    // *=
    Div,    // /=
    Rem,    // %=
    Pow,    // **=
    Shl,    // <<=
    Shr,    // >>=
    BitOr,  // |=
    BitAnd, // &=
    BitXor, // ^=
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallExpr<'src> {
    pub span: Span,
    pub expr: Expression<'src>,
    pub args: Vec<Expression<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CastExpr<'src> {
    pub span: Span,
    pub expr: Expression<'src>,
    pub type_: Type,
}

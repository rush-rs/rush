use std::fmt::{Debug, Display};

use crate::Span;

pub type ParsedProgram<'src> = Program<'src, ()>;
pub type ParsedFunctionDefinition<'src> = FunctionDefinition<'src, ()>;
pub type ParsedBlock<'src> = Block<'src, ()>;
pub type ParsedStatement<'src> = Statement<'src, ()>;
pub type ParsedLetStmt<'src> = LetStmt<'src, ()>;
pub type ParsedReturnStmt<'src> = ReturnStmt<'src, ()>;
pub type ParsedExprStmt<'src> = ExprStmt<'src, ()>;
pub type ParsedExpression<'src> = Expression<'src, ()>;
pub type ParsedIfExpr<'src> = IfExpr<'src, ()>;
pub type ParsedAtom<T> = Atom<T, ()>;
pub type ParsedIdent<'src> = Atom<&'src str, ()>;
pub type ParsedType<'src> = Atom<TypeKind, ()>;
pub type ParsedPrefixExpr<'src> = PrefixExpr<'src, ()>;
pub type ParsedInfixExpr<'src> = InfixExpr<'src, ()>;
pub type ParsedAssignExpr<'src> = AssignExpr<'src, ()>;
pub type ParsedCallExpr<'src> = CallExpr<'src, ()>;
pub type ParsedCastExpr<'src> = CastExpr<'src, ()>;

pub type Type<'src, Annotation> = Atom<TypeKind, Annotation>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeKind {
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

impl Display for TypeKind {
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

#[derive(Debug, Clone, PartialEq)]
pub struct Program<'src, Annotation> {
    pub span: Span,
    pub functions: Vec<FunctionDefinition<'src, Annotation>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDefinition<'src, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub name: Ident<'src, Annotation>,
    pub params: Vec<(Ident<'src, Annotation>, Type<'src, Annotation>)>,
    pub return_type: Type<'src, Annotation>,
    pub block: Block<'src, Annotation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block<'src, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub stmts: Vec<Statement<'src, Annotation>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement<'src, Annotation> {
    Let(LetStmt<'src, Annotation>),
    Return(ReturnStmt<'src, Annotation>),
    Expr(ExprStmt<'src, Annotation>),
}

impl<Annotation> Statement<'_, Annotation> {
    pub fn span(&self) -> Span {
        match self {
            Self::Let(stmt) => stmt.span,
            Self::Return(stmt) => stmt.span,
            Self::Expr(stmt) => stmt.span,
        }
    }

    pub fn annotation(&self) -> &Annotation {
        match self {
            Self::Let(stmt) => &stmt.annotation,
            Self::Return(stmt) => &stmt.annotation,
            Self::Expr(stmt) => &stmt.annotation,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetStmt<'src, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub mutable: bool,
    pub name: Ident<'src, Annotation>,
    pub type_: Option<Type<'src, Annotation>>,
    pub expr: Expression<'src, Annotation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReturnStmt<'src, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub expr: Option<Expression<'src, Annotation>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExprStmt<'src, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub expr: Expression<'src, Annotation>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression<'src, Annotation> {
    Block(Box<Block<'src, Annotation>>),
    If(Box<IfExpr<'src, Annotation>>),
    Int(Atom<i64, Annotation>),
    Float(Atom<f64, Annotation>),
    Bool(Atom<bool, Annotation>),
    Ident(Atom<&'src str, Annotation>),
    Prefix(Box<PrefixExpr<'src, Annotation>>),
    Infix(Box<InfixExpr<'src, Annotation>>),
    Assign(Box<AssignExpr<'src, Annotation>>),
    Call(Box<CallExpr<'src, Annotation>>),
    Cast(Box<CastExpr<'src, Annotation>>),
    Grouped(Atom<Box<Expression<'src, Annotation>>, Annotation>),
}

impl<Annotation> Expression<'_, Annotation> {
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

    pub fn annotation(&self) -> &Annotation {
        match self {
            Self::Block(expr) => &expr.annotation,
            Self::If(expr) => &expr.annotation,
            Self::Int(expr) => &expr.annotation,
            Self::Float(expr) => &expr.annotation,
            Self::Bool(expr) => &expr.annotation,
            Self::Ident(expr) => &expr.annotation,
            Self::Prefix(expr) => &expr.annotation,
            Self::Infix(expr) => &expr.annotation,
            Self::Assign(expr) => &expr.annotation,
            Self::Call(expr) => &expr.annotation,
            Self::Cast(expr) => &expr.annotation,
            Self::Grouped(expr) => &expr.annotation,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfExpr<'src, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub cond: Expression<'src, Annotation>,
    pub then_block: Block<'src, Annotation>,
    pub else_block: Option<Block<'src, Annotation>>,
}

pub type Ident<'src, Annotation> = Atom<&'src str, Annotation>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atom<T, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub value: T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefixExpr<'src, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub op: PrefixOp,
    pub expr: Expression<'src, Annotation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixOp {
    /// !
    Not,
    /// -
    Neg,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InfixExpr<'src, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub lhs: Expression<'src, Annotation>,
    pub op: InfixOp,
    pub rhs: Expression<'src, Annotation>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct AssignExpr<'src, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub assignee: ParsedIdent<'src>,
    pub op: AssignOp,
    pub expr: Expression<'src, Annotation>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct CallExpr<'src, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub expr: Expression<'src, Annotation>,
    pub args: Vec<Expression<'src, Annotation>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CastExpr<'src, Annotation> {
    pub span: Span,
    pub annotation: Annotation,
    pub expr: Expression<'src, Annotation>,
    pub type_: Type<'src, Annotation>,
}

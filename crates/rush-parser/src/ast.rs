use std::fmt::{self, Debug, Display, Formatter};

use crate::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Float,
    Bool,
    Char,
    // TODO: use real implementation when ready
    // Pointer(Box<Type>),
    Pointer,
    Unit,
    /// Internal use only, used for diverging expressions
    Never,
    /// Internal use only, used if typecheck could not determine a type
    Unknown,
}

// TODO: uncomment when ready
/*
impl Type {
    /// Can be called on [`Type::Pointer`] in order to get the inner type of the pointer.
    /// If called on non-pointers, this function will panic
    pub fn deref(self) -> Self {
        match self {
            Type::Pointer(inner) => *inner,
            _ => panic!("called `Type::deref` on a non-pointer variant"),
        }
    }
}
*/

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int => write!(f, "int"),
            Self::Float => write!(f, "float"),
            Self::Bool => write!(f, "bool"),
            Self::Char => write!(f, "char"),
            // TODO: use real implementation
            Self::Pointer => write!(f, "*inner"),
            Self::Unit => write!(f, "()"),
            Self::Never => write!(f, "!"),
            Self::Unknown => write!(f, "{{unknown}}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned<'src, T> {
    pub span: Span<'src>,
    pub inner: T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program<'src> {
    pub span: Span<'src>,
    pub functions: Vec<FunctionDefinition<'src>>,
    pub globals: Vec<LetStmt<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDefinition<'src> {
    pub span: Span<'src>,
    pub name: Spanned<'src, &'src str>,
    pub params: Spanned<'src, Vec<Parameter<'src>>>,
    pub return_type: Spanned<'src, Option<Type>>,
    pub block: Block<'src>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter<'src> {
    pub mutable: bool,
    pub name: Spanned<'src, &'src str>,
    pub type_: Spanned<'src, Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block<'src> {
    pub span: Span<'src>,
    pub stmts: Vec<Statement<'src>>,
    pub expr: Option<Expression<'src>>,
}

impl<'src> Block<'src> {
    /// Returns the span responsible for the block's result type
    pub fn result_span(&self) -> Span<'src> {
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
    Loop(LoopStmt<'src>),
    While(WhileStmt<'src>),
    For(ForStmt<'src>),
    Break(BreakStmt<'src>),
    Continue(ContinueStmt<'src>),
    Expr(ExprStmt<'src>),
}

impl<'src> Statement<'src> {
    pub fn span(&self) -> Span<'src> {
        match self {
            Self::Let(stmt) => stmt.span,
            Self::Return(stmt) => stmt.span,
            Self::Loop(stmt) => stmt.span,
            Self::While(stmt) => stmt.span,
            Self::For(stmt) => stmt.span,
            Self::Break(stmt) => stmt.span,
            Self::Continue(stmt) => stmt.span,
            Self::Expr(stmt) => stmt.span,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetStmt<'src> {
    pub span: Span<'src>,
    pub mutable: bool,
    pub name: Spanned<'src, &'src str>,
    pub type_: Option<Spanned<'src, Type>>,
    pub expr: Expression<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReturnStmt<'src> {
    pub span: Span<'src>,
    pub expr: Option<Expression<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopStmt<'src> {
    pub span: Span<'src>,
    pub block: Block<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhileStmt<'src> {
    pub span: Span<'src>,
    pub cond: Expression<'src>,
    pub block: Block<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForStmt<'src> {
    pub span: Span<'src>,
    pub ident: Spanned<'src, &'src str>,
    pub initializer: Expression<'src>,
    pub cond: Expression<'src>,
    pub update: Expression<'src>,
    pub block: Block<'src>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BreakStmt<'src> {
    pub span: Span<'src>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinueStmt<'src> {
    pub span: Span<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExprStmt<'src> {
    pub span: Span<'src>,
    pub expr: Expression<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression<'src> {
    Block(Box<Block<'src>>),
    If(Box<IfExpr<'src>>),
    Int(Spanned<'src, i64>),
    Float(Spanned<'src, f64>),
    Bool(Spanned<'src, bool>),
    Char(Spanned<'src, u8>),
    Ident(Spanned<'src, &'src str>),
    Ref(Spanned<'src, &'src str>),
    Deref(Spanned<'src, &'src str>),
    Prefix(Box<PrefixExpr<'src>>),
    Infix(Box<InfixExpr<'src>>),
    Assign(Box<AssignExpr<'src>>),
    Call(Box<CallExpr<'src>>),
    Cast(Box<CastExpr<'src>>),
    Grouped(Spanned<'src, Box<Expression<'src>>>),
}

impl<'src> Expression<'src> {
    pub fn span(&self) -> Span<'src> {
        match self {
            Self::Block(expr) => expr.span,
            Self::If(expr) => expr.span,
            Self::Int(expr) => expr.span,
            Self::Float(expr) => expr.span,
            Self::Bool(expr) => expr.span,
            Self::Char(expr) => expr.span,
            Self::Ident(expr) => expr.span,
            Self::Ref(expr) => expr.span,
            Self::Deref(expr) => expr.span,
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
    pub span: Span<'src>,
    pub cond: Expression<'src>,
    pub then_block: Block<'src>,
    pub else_block: Option<Block<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefixExpr<'src> {
    pub span: Span<'src>,
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
    pub span: Span<'src>,
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

impl From<AssignOp> for InfixOp {
    fn from(src: AssignOp) -> Self {
        match src {
            AssignOp::Plus => InfixOp::Plus,
            AssignOp::Minus => InfixOp::Minus,
            AssignOp::Mul => InfixOp::Mul,
            AssignOp::Div => InfixOp::Div,
            AssignOp::Shl => InfixOp::Shl,
            AssignOp::Shr => InfixOp::Shr,
            AssignOp::Rem => InfixOp::Rem,
            AssignOp::Pow => InfixOp::Pow,
            AssignOp::BitOr => InfixOp::BitOr,
            AssignOp::BitAnd => InfixOp::BitAnd,
            AssignOp::BitXor => Self::BitXor,
            AssignOp::Basic => panic!("cannot convert assign op basic to infix op"),
        }
    }
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
    pub span: Span<'src>,
    pub assignee: Spanned<'src, &'src str>,
    pub assignee_is_ptr: bool,
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
    pub span: Span<'src>,
    pub func: Spanned<'src, &'src str>,
    pub args: Vec<Expression<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CastExpr<'src> {
    pub span: Span<'src>,
    pub expr: Expression<'src>,
    pub type_: Spanned<'src, Type>,
}

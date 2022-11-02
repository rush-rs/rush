use crate::Span;

#[derive(Debug, Clone)]
pub struct Program<'src> {
    pub span: Span,
    pub functions: Vec<FunctionDefinition<'src>>,
}

#[derive(Debug, Clone)]
pub enum Type {
    Int,
    Float,
    Bool,
    Char,
    Unit,
}

#[derive(Debug, Clone)]
pub struct FunctionDefinition<'src> {
    pub span: Span,
    pub name: &'src str,
    pub params: Vec<(&'src str, Type)>,
    pub return_type: Type,
    pub block: Block<'src>,
}

#[derive(Debug, Clone)]
pub struct Block<'src> {
    pub span: Span,
    pub stmts: Vec<Statement<'src>>,
}

#[derive(Debug, Clone)]
pub enum Statement<'src> {
    Let(LetStmt<'src>),
    Return(ReturnStmt<'src>),
    Expr(ExprStmt<'src>),
}

#[derive(Debug, Clone)]
pub struct LetStmt<'src> {
    pub span: Span,
    pub mutable: bool,
    pub name: &'src str,
    pub expr: Expression<'src>,
}

#[derive(Debug, Clone)]
pub struct ReturnStmt<'src> {
    pub span: Span,
    pub expr: Option<Expression<'src>>,
}

#[derive(Debug, Clone)]
pub struct ExprStmt<'src> {
    pub span: Span,
    pub expr: Expression<'src>,
}

#[derive(Debug, Clone)]
pub enum Expression<'src> {
    Block(Box<Block<'src>>),
    If(Box<IfExpr<'src>>),
    Int(i64),
    Float(f64),
    Bool(bool),
    Ident(&'src str),
    Prefix(Box<PrefixExpr<'src>>),
    Infix(Box<InfixExpr<'src>>),
    Call(Box<CallExpr<'src>>),
    Grouped(Box<Expression<'src>>),
}

#[derive(Debug, Clone)]
pub struct IfExpr<'src> {
    pub span: Span,
    pub cond: Expression<'src>,
    pub then_block: Block<'src>,
    pub else_block: Option<Block<'src>>,
}

#[derive(Debug, Clone)]
pub struct PrefixExpr<'src> {
    pub span: Span,
    pub op: PrefixOp,
    pub expr: Expression<'src>,
}

#[derive(Debug, Clone)]
pub enum PrefixOp {
    Not, // !
    Neg, // -
}

#[derive(Debug, Clone)]
pub struct InfixExpr<'src> {
    pub span: Span,
    pub left: Expression<'src>,
    pub op: InfixOp,
    pub right: Expression<'src>,
}

#[derive(Debug, Clone)]
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

    Assign,       // =
    PlusAssign,   // +=
    MinusAssign,  // -=
    MulAssign,    // *=
    DivAssign,    // /=
    RemAssign,    // %=
    PowAssign,    // **=
    ShlAssign,    // <<=
    ShrAssign,    // >>=
    BitOrAssign,  // |=
    BitAndAssign, // &=
    BitXorAssign, // ^=
}

#[derive(Debug, Clone)]
pub struct CallExpr<'src> {
    pub span: Span,
    pub expr: Expression<'src>,
    pub args: Vec<Expression<'src>>,
}

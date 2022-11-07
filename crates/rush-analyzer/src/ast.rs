use rush_parser::ast::{AssignOp, InfixOp, PrefixOp, Type};

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedProgram<'src> {
    pub functions: Vec<AnalyzedFunctionDefinition<'src>>,
    pub main_fn: AnalyzedBlock<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedFunctionDefinition<'src> {
    pub used: bool,
    pub name: &'src str,
    pub params: Vec<(&'src str, Type)>,
    pub return_type: Type,
    pub block: AnalyzedBlock<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedBlock<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub stmts: Vec<AnalyzedStatement<'src>>,
    pub expr: Option<AnalyzedExpression<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnalyzedStatement<'src> {
    Let(AnalyzedLetStmt<'src>),
    Return(AnalyzedReturnStmt<'src>),
    Expr(AnalyzedExpression<'src>),
}

impl AnalyzedStatement<'_> {
    pub fn result_type(&self) -> Type {
        match self {
            Self::Let(_) => Type::Unit,
            Self::Return(_) => Type::Never,
            Self::Expr(expr) => expr.result_type(),
        }
    }

    pub fn constant(&self) -> bool {
        match self {
            Self::Let(_) => false,
            Self::Return(_) => false,
            Self::Expr(expr) => expr.constant(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedLetStmt<'src> {
    pub name: &'src str,
    pub expr: AnalyzedExpression<'src>,
}

pub type AnalyzedReturnStmt<'src> = Option<AnalyzedExpression<'src>>;

#[derive(Debug, Clone, PartialEq)]
pub enum AnalyzedExpression<'src> {
    Block(Box<AnalyzedBlock<'src>>),
    If(Box<AnalyzedIfExpr<'src>>),
    Int(i64),
    Float(f64),
    Bool(bool),
    Char(u8),
    Ident(AnalyzedIdentExpr<'src>),
    Prefix(Box<AnalyzedPrefixExpr<'src>>),
    Infix(Box<AnalyzedInfixExpr<'src>>),
    Assign(Box<AnalyzedAssignExpr<'src>>),
    Call(Box<AnalyzedCallExpr<'src>>),
    Cast(Box<AnalyzedCastExpr<'src>>),
    Grouped(Box<AnalyzedExpression<'src>>),
}

impl AnalyzedExpression<'_> {
    pub fn result_type(&self) -> Type {
        match self {
            Self::Block(expr) => expr.result_type,
            Self::Int(_) => Type::Int,
            Self::Float(_) => Type::Float,
            Self::Bool(_) => Type::Bool,
            Self::Char(_) => Type::Char,
            Self::Ident(expr) => expr.result_type,
            Self::If(expr) => expr.result_type,
            Self::Prefix(expr) => expr.result_type,
            Self::Infix(expr) => expr.result_type,
            Self::Assign(expr) => expr.result_type,
            Self::Call(expr) => expr.result_type,
            Self::Cast(expr) => expr.result_type,
            Self::Grouped(expr) => expr.result_type(),
        }
    }

    pub fn constant(&self) -> bool {
        match self {
            Self::Block(expr) => expr.constant,
            Self::Int(_) => true,
            Self::Float(_) => true,
            Self::Bool(_) => true,
            Self::Char(_) => true,
            Self::Ident(_) => false,
            Self::If(expr) => expr.constant,
            Self::Prefix(expr) => expr.constant,
            Self::Infix(expr) => expr.constant,
            Self::Assign(_) => false,
            Self::Call(_) => false,
            Self::Cast(expr) => expr.constant,
            Self::Grouped(expr) => expr.constant(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedIfExpr<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub cond: AnalyzedExpression<'src>,
    pub then_block: AnalyzedBlock<'src>,
    pub else_block: Option<AnalyzedBlock<'src>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyzedIdentExpr<'src> {
    pub result_type: Type,
    pub ident: &'src str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedPrefixExpr<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub op: PrefixOp,
    pub expr: AnalyzedExpression<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedInfixExpr<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub lhs: AnalyzedExpression<'src>,
    pub op: InfixOp,
    pub rhs: AnalyzedExpression<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedAssignExpr<'src> {
    pub result_type: Type,
    pub assignee: &'src str,
    pub op: AssignOp,
    pub expr: AnalyzedExpression<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedCallExpr<'src> {
    pub result_type: Type,
    pub func: &'src str,
    pub args: Vec<AnalyzedExpression<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedCastExpr<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub expr: AnalyzedExpression<'src>,
    pub type_: Type,
}

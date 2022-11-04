use rush_parser::ast::{AssignOp, InfixOp, PrefixOp, Type};

pub type AnalysedProgram<'src> = Vec<AnalysedFunctionDefinition<'src>>;

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysedFunctionDefinition<'src> {
    pub name: &'src str,
    pub params: Vec<(&'src str, Type)>,
    pub return_type: Type,
    pub block: AnalysedBlock<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysedBlock<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub stmts: Vec<AnalysedStatement<'src>>,
    pub expr: Option<AnalysedExpression<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnalysedStatement<'src> {
    Let(AnalysedLetStmt<'src>),
    Return(AnalysedReturnStmt<'src>),
    Expr(AnalysedExpression<'src>),
}

impl AnalysedStatement<'_> {
    pub fn result_type(&self) -> Type {
        match self {
            Self::Let(_) => Type::Unit,
            Self::Return(_) => Type::Unit,
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
pub struct AnalysedLetStmt<'src> {
    pub mutable: bool,
    pub name: &'src str,
    pub expr: AnalysedExpression<'src>,
}

pub type AnalysedReturnStmt<'src> = Option<AnalysedExpression<'src>>;

#[derive(Debug, Clone, PartialEq)]
pub enum AnalysedExpression<'src> {
    Block(Box<AnalysedBlock<'src>>),
    If(Box<AnalysedIfExpr<'src>>),
    Int(i64),
    Float(f64),
    Bool(bool),
    Ident(IdentExpr<'src>),
    Prefix(Box<AnalysedPrefixExpr<'src>>),
    Infix(Box<AnalysedInfixExpr<'src>>),
    Assign(Box<AnalysedAssignExpr<'src>>),
    Call(Box<AnalysedCallExpr<'src>>),
    Cast(Box<AnalysedCastExpr<'src>>),
    Grouped(Box<AnalysedExpression<'src>>),
}

impl AnalysedExpression<'_> {
    pub fn result_type(&self) -> Type {
        match self {
            Self::Block(expr) => expr.result_type,
            Self::Int(_) => Type::Int,
            Self::Float(_) => Type::Float,
            Self::Bool(_) => Type::Bool,
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
            Self::Ident(_) => false,
            Self::If(expr) => expr.constant,
            Self::Prefix(expr) => expr.constant,
            Self::Infix(expr) => expr.constant,
            Self::Assign(expr) => expr.constant,
            Self::Call(expr) => expr.constant,
            Self::Cast(expr) => expr.constant,
            Self::Grouped(expr) => expr.constant(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysedIfExpr<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub cond: AnalysedExpression<'src>,
    pub then_block: AnalysedBlock<'src>,
    pub else_block: Option<AnalysedBlock<'src>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentExpr<'src> {
    pub result_type: Type,
    pub ident: &'src str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysedPrefixExpr<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub op: PrefixOp,
    pub expr: AnalysedExpression<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysedInfixExpr<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub lhs: AnalysedExpression<'src>,
    pub op: InfixOp,
    pub rhs: AnalysedExpression<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysedAssignExpr<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub assignee: &'src str,
    pub op: AssignOp,
    pub expr: AnalysedExpression<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysedCallExpr<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub expr: AnalysedExpression<'src>,
    pub args: Vec<AnalysedExpression<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysedCastExpr<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub expr: AnalysedExpression<'src>,
    pub type_: Type,
}

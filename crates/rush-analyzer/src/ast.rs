use std::collections::HashSet;

use rush_parser::ast::{AssignOp, InfixOp, PrefixOp, Type};

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedProgram<'src> {
    pub globals: Vec<AnalyzedLetStmt<'src>>,
    pub functions: Vec<AnalyzedFunctionDefinition<'src>>,
    pub main_fn: AnalyzedBlock<'src>,
    pub used_builtins: HashSet<&'src str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedFunctionDefinition<'src> {
    pub used: bool,
    pub name: &'src str,
    pub params: Vec<AnalyzedParameter<'src>>,
    pub return_type: Type,
    pub block: AnalyzedBlock<'src>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyzedParameter<'src> {
    pub mutable: bool,
    pub name: &'src str,
    pub type_: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedBlock<'src> {
    pub result_type: Type,
    pub stmts: Vec<AnalyzedStatement<'src>>,
    pub expr: Option<AnalyzedExpression<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnalyzedStatement<'src> {
    Let(AnalyzedLetStmt<'src>),
    Return(AnalyzedReturnStmt<'src>),
    Loop(AnalyzedLoopStmt<'src>),
    While(AnalyzedWhileStmt<'src>),
    For(AnalyzedForStmt<'src>),
    Break,
    Continue,
    Expr(AnalyzedExpression<'src>),
}

impl AnalyzedStatement<'_> {
    pub fn result_type(&self) -> Type {
        match self {
            Self::Let(_) => Type::Unit,
            Self::Return(_) => Type::Never,
            Self::Loop(node) => match node.never_terminates {
                true => Type::Never,
                false => Type::Unit,
            }, // Used for detecting never-ending loops
            Self::While(node) => match node.never_terminates {
                true => Type::Never,
                false => Type::Unit,
            }, // Used for detecting never-ending loops
            Self::For(node) => match node.never_terminates {
                true => Type::Never,
                false => Type::Unit,
            }, // Used for detecting never-ending loops
            Self::Break => Type::Never,
            Self::Continue => Type::Never,
            Self::Expr(expr) => expr.result_type(),
        }
    }

    pub fn constant(&self) -> bool {
        match self {
            Self::Expr(expr) => expr.constant(),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedLetStmt<'src> {
    pub name: &'src str,
    pub expr: AnalyzedExpression<'src>,
    pub mutable: bool,
    pub used: bool,
}

pub type AnalyzedReturnStmt<'src> = Option<AnalyzedExpression<'src>>;

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedLoopStmt<'src> {
    pub block: AnalyzedBlock<'src>,
    // specifies the allocations performed by the current loop
    pub allocations: Vec<(&'src str, Type)>,
    pub never_terminates: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedWhileStmt<'src> {
    pub cond: AnalyzedExpression<'src>,
    pub block: AnalyzedBlock<'src>,
    // specifies the allocations performed by the current loop
    pub allocations: Vec<(&'src str, Type)>,
    pub never_terminates: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedForStmt<'src> {
    pub ident: &'src str,
    pub initializer: AnalyzedExpression<'src>,
    pub cond: AnalyzedExpression<'src>,
    pub update: AnalyzedExpression<'src>,
    pub block: AnalyzedBlock<'src>,
    // specifies the allocations performed by the current loop
    pub allocations: Vec<(&'src str, Type)>,
    pub never_terminates: bool,
}

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
        matches!(
            self,
            Self::Int(_) | Self::Float(_) | Self::Bool(_) | Self::Char(_)
        )
    }

    pub fn as_constant(&self) -> Option<Self> {
        match self {
            AnalyzedExpression::Int(_)
            | AnalyzedExpression::Float(_)
            | AnalyzedExpression::Bool(_)
            | AnalyzedExpression::Char(_) => {
                // this clone is cheap, as inner values of these variants all impl `Copy`
                Some(self.clone())
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedIfExpr<'src> {
    pub result_type: Type,
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
    pub op: PrefixOp,
    pub expr: AnalyzedExpression<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzedInfixExpr<'src> {
    pub result_type: Type,
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
    pub expr: AnalyzedExpression<'src>,
    pub type_: Type,
}

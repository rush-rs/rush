use rush_parser::ast::*;

pub type AnnotatedProgram<'src> = Program<'src, AnnotatedStatement<'src>>;
pub type AnnotatedFunctionDefinition<'src> = FunctionDefinition<'src, AnnotatedStatement<'src>>;
pub type AnnotatedBlock<'src> = Block<AnnotatedStatement<'src>>;

#[derive(Debug, Clone, PartialEq)]
pub struct AnnotatedStatement<'src> {
    pub result_type: Type,
    pub stmt: Statement<'src, AnnotatedExpression<'src>>,
}

pub type AnnotatedLetStmt<'src> = LetStmt<'src, AnnotatedExpression<'src>>;
pub type AnnotatedReturnStmt<'src> = ReturnStmt<AnnotatedExpression<'src>>;
pub type AnnotatedExprStmt<'src> = ExprStmt<AnnotatedExpression<'src>>;

#[derive(Debug, Clone, PartialEq)]
pub struct AnnotatedExpression<'src> {
    pub result_type: Type,
    pub constant: bool,
    pub expr: Expression<'src, AnnotatedStatement<'src>, AnnotatedExpression<'src>>,
}

pub type AnnotatedIfExpr<'src> = IfExpr<AnnotatedStatement<'src>, AnnotatedExpression<'src>>;
pub type AnnotatedPrefixExpr<'src> = PrefixExpr<AnnotatedExpression<'src>>;
pub type AnnotatedInfixExpr<'src> = InfixExpr<AnnotatedExpression<'src>>;
pub type AnnotatedAssignExpr<'src> = AssignExpr<'src, AnnotatedExpression<'src>>;
pub type AnnotatedCallExpr<'src> = CallExpr<AnnotatedExpression<'src>>;
pub type AnnotatedCastExpr<'src> = CastExpr<AnnotatedExpression<'src>>;

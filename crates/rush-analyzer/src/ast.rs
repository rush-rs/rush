use rush_parser::ast::*;

pub struct Annotation {
    pub result_type: TypeKind,
    pub constant: bool,
}

impl Annotation {
    pub fn new(result_type: TypeKind, constant: bool) -> Self {
        Self {
            result_type,
            constant,
        }
    }
}

pub type AnnotatedProgram<'src> = Program<'src, Annotation>;
pub type AnnotatedFunctionDefinition<'src> = FunctionDefinition<'src, Annotation>;
pub type AnnotatedBlock<'src> = Block<'src, Annotation>;
pub type AnnotatedStatement<'src> = Statement<'src, Annotation>;
pub type AnnotatedLetStmt<'src> = LetStmt<'src, Annotation>;
pub type AnnotatedReturnStmt<'src> = ReturnStmt<'src, Annotation>;
pub type AnnotatedExprStmt<'src> = ExprStmt<'src, Annotation>;
pub type AnnotatedExpression<'src> = Expression<'src, Annotation>;
pub type AnnotatedIfExpr<'src> = IfExpr<'src, Annotation>;
pub type AnnotatedPrefixExpr<'src> = PrefixExpr<'src, Annotation>;
pub type AnnotatedInfixExpr<'src> = InfixExpr<'src, Annotation>;
pub type AnnotatedAssignExpr<'src> = AssignExpr<'src, Annotation>;
pub type AnnotatedCallExpr<'src> = CallExpr<'src, Annotation>;
pub type AnnotatedCastExpr<'src> = CastExpr<'src, Annotation>;
pub type AnnotatedIdent<'src> = Atom<&'src str, Annotation>;
pub type AnnotatedType<'src> = Atom<TypeKind, Annotation>;

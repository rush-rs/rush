use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use rush_parser::{ast::*, Span};

use crate::{ast::*, Diagnostic, DiagnosticLevel, ErrorKind};

#[derive(Default, Debug)]
pub struct Analyzer<'src> {
    pub functions: HashMap<&'src str, Function<'src>>,
    scope: Option<Scope<'src>>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug)]
pub struct Function<'src> {
    pub ident: Spanned<&'src str>,
    pub params: Spanned<Vec<(Spanned<&'src str>, Spanned<Type>)>>,
    pub return_type: Spanned<Option<Type>>,
    pub used: bool,
}

#[derive(Debug)]
pub struct Scope<'src> {
    pub fn_name: &'src str,
    pub vars: HashMap<&'src str, Variable>,
}

#[derive(Debug)]
pub struct Variable {
    pub type_: Type,
    pub span: Span,
    pub used: bool,
    pub mutable: bool,
}

impl<'src> Analyzer<'src> {
    /// Creates a new [`Analyzer`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new diagnostic with the `Hint` level
    fn hint(&mut self, message: impl Into<Cow<'static, str>>, span: Span) {
        self.diagnostics.push(Diagnostic::new(
            DiagnosticLevel::Hint,
            message,
            vec![],
            span,
        ))
    }

    /// Adds a new diagnostic with the `Info` level
    fn info(
        &mut self,
        message: impl Into<Cow<'static, str>>,
        notes: Vec<Cow<'static, str>>,
        span: Span,
    ) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Info, message, notes, span))
    }

    /// Adds a new diagnostic with the `Warning` level
    fn warn(
        &mut self,
        message: impl Into<Cow<'static, str>>,
        notes: Vec<Cow<'static, str>>,
        span: Span,
    ) {
        self.diagnostics.push(Diagnostic::new(
            DiagnosticLevel::Warning,
            message,
            notes,
            span,
        ))
    }

    /// Adds a new diagnostic with the `Error` level using the specified error kind
    fn error(
        &mut self,
        kind: ErrorKind,
        message: impl Into<Cow<'static, str>>,
        notes: Vec<Cow<'static, str>>,
        span: Span,
    ) {
        self.diagnostics.push(Diagnostic::new(
            DiagnosticLevel::Error(kind),
            message,
            notes,
            span,
        ))
    }

    /// Analyzes a parsed AST and returns an analyzed AST whilst emitting diagnostics
    pub fn analyze(
        mut self,
        program: Program<'src>,
    ) -> Result<(AnalyzedProgram<'src>, Vec<Diagnostic>), Vec<Diagnostic>> {
        let mut functions = vec![];
        let mut main_fn = None;

        for func in program.functions {
            let func = self.visit_function_definition(func);
            match func.name {
                "main" => {
                    main_fn = Some(func.block);
                }
                _ => functions.push(func),
            }
        }

        // check if there are any unused functions
        let unused_funcs: Vec<_> = self
            .functions
            .values()
            .filter(|func| {
                func.ident.inner != "main" && !func.ident.inner.starts_with('_') && !func.used
            })
            .map(|func| {
                // set used = false in tree
                functions
                    .iter_mut()
                    .find(|func_def| func_def.name == func.ident.inner)
                    .expect("every unused function is defined")
                    .used = false;

                func.ident.clone()
            })
            .collect();

        for ident in unused_funcs {
            self.warn(
                format!("function `{}` is never called", ident.inner),
                vec![format!(
                    "if this is intentional, change the name to `_{}` to hide this warning",
                    ident.inner,
                )
                .into()],
                ident.span,
            )
        }

        match main_fn {
            Some(main_fn) => Ok((AnalyzedProgram { functions, main_fn }, self.diagnostics)),
            None => {
                self.error(
                    ErrorKind::Semantic,
                    "missing `main` function",
                    vec![
                        "the `main` function can be implemented like this: `fn main() { ... }`"
                            .into(),
                    ],
                    Span::default(),
                );
                Err(self.diagnostics)
            }
        }
    }

    /// Removes the current scope of the function and checks whether the
    /// variables in the scope have been used.
    fn drop_scope(&mut self) {
        // consume / drop the scope
        let scope = self
            .scope
            .take()
            .expect("drop_scope should only be called from a scope");

        // analyze its values for their use
        for (name, var) in scope.vars {
            // allow unused values if they start with `_`
            if !name.starts_with('_') && !var.used {
                self.warn(
                    format!("unused variable `{}`", name),
                    vec![format!(
                        "if this is intentional, change the name to `_{name}` to hide this warning"
                    )
                    .into()],
                    var.span,
                );
            }
        }
    }

    /// Unwrap the current scope
    fn scope(&self) -> &Scope<'src> {
        self.scope
            .as_ref()
            .expect("statements and expressions only exist in function bodies")
    }

    /// Unwrap the current scope mutably
    fn scope_mut(&mut self) -> &mut Scope<'src> {
        self.scope
            .as_mut()
            .expect("statements and expressions only exist in function bodies")
    }

    fn warn_unreachable_expr(&mut self, unreachable_span: Span, causing_span: Span) {
        self.warn("unreachable expression", vec![], unreachable_span);
        self.hint(
            "any code following this expression is unreachable",
            causing_span,
        );
    }

    fn visit_function_definition(
        &mut self,
        node: FunctionDefinition<'src>,
    ) -> AnalyzedFunctionDefinition<'src> {
        // check for duplicate function names
        if self.functions.contains_key(node.name.inner) {
            self.error(
                ErrorKind::Semantic,
                format!("duplicate function definition `{}`", node.name.inner),
                vec![],
                node.name.span,
            );
        }

        // check if the function is the main function
        let is_main_fn = node.name.inner == "main";

        if is_main_fn {
            // the main function must have 0 parameters
            if !node.params.inner.is_empty() {
                self.error(
                    ErrorKind::Semantic,
                    format!(
                        "the `main` function must have 0 parameters, however {} {} defined",
                        node.params.inner.len(),
                        if node.params.inner.len() == 1 {
                            "is"
                        } else {
                            "are"
                        },
                    ),
                    vec!["remove the parameters: `fn main() { ... }`".into()],
                    node.params.span,
                )
            }

            // the main function must return `()`
            if let Some(return_type) = node.return_type.inner {
                if return_type != Type::Unit {
                    self.error(
                        ErrorKind::Semantic,
                        format!(
                            "the `main` function's return type must be `()`, but is declared as `{}`",
                            return_type,
                        ),
                        vec!["remove the return type: `fn main() { ... }`".into()],
                        node.return_type.span,
                    )
                }
            }
        }

        // info for explicit unit return type
        if node.return_type.inner == Some(Type::Unit) {
            self.info(
                "unnecessary explicit unit return type",
                vec![
                    "functions implicitly return `()` by default".into(),
                    format!(
                        "remove the explicit type: `fn {}(...) {{ ... }}`",
                        node.name.inner,
                    )
                    .into(),
                ],
                node.return_type.span,
            );
        }

        let mut scope_vars = HashMap::new();

        // check the function parameters
        let mut params = vec![];
        let mut param_names = HashSet::new();

        // only analyze parameters if this is not the main function
        if !is_main_fn {
            for (ident, type_) in node.params.inner {
                // check for duplicate function parameters
                if !param_names.insert(ident.inner) {
                    self.error(
                        ErrorKind::Semantic,
                        format!("duplicate parameter name `{}`", ident.inner),
                        vec![],
                        ident.span,
                    );
                }
                scope_vars.insert(
                    ident.inner,
                    Variable {
                        type_: type_.inner,
                        span: ident.span,
                        used: false,
                        // TODO: maybe allow `mut` params
                        mutable: false,
                    },
                );
                params.push((ident, type_));
            }
        }

        // add the function to the analyzer's function map
        self.functions.insert(
            node.name.inner,
            Function {
                params: Spanned {
                    span: node.params.span,
                    inner: params.clone(),
                },
                return_type: node.return_type.clone(),
                used: false,
                ident: node.name.clone(),
            },
        );

        // set the scope to a new blank scope
        self.scope = Some(Scope {
            fn_name: node.name.inner,
            vars: scope_vars,
        });

        // analyze the function body
        let block_result_span = node.block.result_span();
        let block = self.visit_block(node.block);

        // check that the block results in the expected type
        if block.result_type != node.return_type.inner.unwrap_or(Type::Unit)
            // unknown and never types are tolerated
            && !matches!(block.result_type, Type::Unknown | Type::Never)
        {
            self.error(
                ErrorKind::Type,
                format!(
                    "mismatched types: expected `{}`, found `{}`",
                    node.return_type.inner.unwrap_or(Type::Unit),
                    block.result_type,
                ),
                match node.return_type.inner.is_some() {
                    true => vec![],
                    false => vec![format!(
                        "specify a function return type like this: `fn {}(...) -> {} {{ ... }}`",
                        node.name.inner, block.result_type,
                    )
                    .into()],
                },
                block_result_span,
            );
            self.hint(
                match node.return_type.inner.is_some() {
                    true => "function return type defined here",
                    false => "no explicit return type specified",
                },
                node.return_type.span,
            );
        }

        let params_without_spans = params
            .iter()
            .map(|(ident, type_)| (ident.inner, type_.inner))
            .collect();

        // drop the scope when finished
        self.drop_scope();

        AnalyzedFunctionDefinition {
            used: true, // is modified in Self::analyze()
            name: node.name.inner,
            params: params_without_spans,
            return_type: node.return_type.inner.unwrap_or(Type::Unit),
            block,
        }
    }

    fn visit_block(&mut self, node: Block<'src>) -> AnalyzedBlock<'src> {
        let mut stmts = vec![];

        let mut never_type_span = None;
        let mut warned_unreachable = false;

        for stmt in node.stmts {
            if let Some(span) = never_type_span {
                if !warned_unreachable {
                    self.warn("unreachable statement", vec![], stmt.span());
                    self.hint("any code following this statement is unreachable", span);
                    warned_unreachable = true;
                }
            }
            let stmt_span = stmt.span();
            let stmt = self.visit_statement(stmt);
            if stmt.result_type() == Type::Never {
                never_type_span = Some(stmt_span);
            }
            stmts.push(stmt);
        }

        // possibly mark trailing expression as unreachable
        if let (Some(expr), Some(span), false) = (&node.expr, never_type_span, warned_unreachable) {
            self.warn("unreachable expression", vec![], expr.span());
            self.hint("any code following this statement is unreachable", span);
        }

        // analyze the expression
        let expr = node.expr.map(|expr| self.visit_expression(expr));

        // result type is `!` when any statement had type `!`, otherwise the type of the expr
        let result_type = match never_type_span {
            Some(_) => Type::Never,
            None => expr.as_ref().map_or(Type::Unit, |expr| expr.result_type()),
        };
        let constant = expr.as_ref().map_or(true, |expr| expr.constant()) && stmts.is_empty();

        AnalyzedBlock {
            result_type,
            constant,
            stmts,
            expr,
        }
    }

    fn visit_statement(&mut self, node: Statement<'src>) -> AnalyzedStatement<'src> {
        match node {
            Statement::Let(node) => self.visit_let_stmt(node),
            Statement::Return(node) => self.visit_return_stmt(node),
            Statement::Expr(node) => AnalyzedStatement::Expr(self.visit_expression(node.expr)),
        }
    }

    fn visit_let_stmt(&mut self, node: LetStmt<'src>) -> AnalyzedStatement<'src> {
        // save the expression's span for later use
        let expr_span = node.expr.span();

        // analyze the right hand side first
        let expr = self.visit_expression(node.expr);

        // check if the optional type conflicts with the rhs
        // if the type of the rhs is unknown, do not return an error
        if let Some(declared) = &node.type_ {
            if declared.inner != expr.result_type() && expr.result_type() != Type::Unknown {
                self.error(
                    ErrorKind::Type,
                    format!(
                        "mismatched types: expected `{}`, found `{}`",
                        declared.inner,
                        expr.result_type(),
                    ),
                    vec![],
                    expr_span,
                );
                self.hint("expected due to this", declared.span);
            }
        }

        // insert and do additional checks if variable is shadowed
        if let Some(old) = self.scope_mut().vars.insert(
            node.name.inner,
            Variable {
                type_: node.type_.map_or(expr.result_type(), |type_| type_.inner),
                span: node.name.span,
                used: false,
                mutable: node.mutable,
            },
        ) {
            // a previous variable is shadowed by this declaration, analyze its use
            if !old.used && !node.name.inner.starts_with('_') {
                self.warn(
                    format!("unused variable `{}`", node.name.inner),
                    vec![format!(
                        "if this is intentional, change the name to `_{}` to hide this warning",
                        node.name.inner
                    )
                    .into()],
                    old.span,
                );
                self.hint(
                    format!("variable `{}` shadowed here", node.name.inner),
                    node.name.span,
                );
            }
        }

        AnalyzedStatement::Let(AnalyzedLetStmt {
            name: node.name.inner,
            expr,
        })
    }

    fn visit_return_stmt(&mut self, node: ReturnStmt<'src>) -> AnalyzedStatement<'src> {
        // if there is an expression, visit it
        let expr = node.expr.map(|expr| self.visit_expression(expr));

        // get the return type based on the expr (Unit as fallback)
        let return_type = expr.as_ref().map_or(Type::Unit, |expr| expr.result_type());

        let curr_fn = self
            .functions
            .get(self.scope().fn_name)
            .expect("a scope's function always exists");

        // test if the return type is correct
        if curr_fn.return_type.inner.unwrap_or(Type::Unit) != return_type
            && return_type != Type::Unknown
        {
            let fn_type_span = curr_fn.return_type.span;
            let fn_type_explicit = curr_fn.return_type.inner.is_some();

            self.error(
                ErrorKind::Type,
                format!(
                    "mismatched types: expected `{}`, found `{}`",
                    curr_fn.return_type.inner.unwrap_or(Type::Unit),
                    return_type
                ),
                vec![],
                node.span,
            );
            self.hint(
                match fn_type_explicit {
                    true => "function return type defined here",
                    false => "no explicit return type specified",
                },
                fn_type_span,
            );
        }

        AnalyzedStatement::Return(expr)
    }

    fn visit_expression(&mut self, node: Expression<'src>) -> AnalyzedExpression<'src> {
        match node {
            Expression::Block(node) => AnalyzedExpression::Block(self.visit_block(*node).into()),
            Expression::If(node) => self.visit_if_expression(*node),
            Expression::Int(node) => AnalyzedExpression::Int(node.inner),
            Expression::Float(node) => AnalyzedExpression::Float(node.inner),
            Expression::Bool(node) => AnalyzedExpression::Bool(node.inner),
            Expression::Char(node) => AnalyzedExpression::Char(node.inner),
            Expression::Ident(node) => self.visit_ident_expr(node),
            Expression::Prefix(node) => self.visit_prefix_expr(*node),
            Expression::Infix(node) => self.visit_infix_expr(*node),
            Expression::Assign(node) => self.visit_assign_expr(*node),
            Expression::Call(node) => self.visit_call_expr(*node),
            Expression::Cast(node) => self.visit_cast_expr(*node),
            Expression::Grouped(node) => {
                AnalyzedExpression::Grouped(self.visit_expression(*node.inner).into())
            }
        }
    }

    fn visit_if_expression(&mut self, node: IfExpr<'src>) -> AnalyzedExpression<'src> {
        let cond_span = node.cond.span();
        let cond = self.visit_expression(node.cond);

        // check that the condition is of type bool
        if !matches!(cond.result_type(), Type::Bool | Type::Unknown) {
            self.error(
                ErrorKind::Type,
                format!(
                    "expected value of type `bool`, found `{}`",
                    cond.result_type()
                ),
                vec!["a condition must have the type `bool`".into()],
                cond_span,
            )
        } else {
            // check that the condition is non-constant
            if cond.constant() {
                self.warn(
                    "redundant if expression: condition is constant",
                    vec![],
                    cond_span,
                )
            }
        }

        // analyze then_block
        let then_result_span = node.then_block.result_span();
        let then_block = self.visit_block(node.then_block);

        // analyze else_block if it exists
        let result_type;
        let else_block = match node.else_block {
            Some(else_block) => {
                let else_result_span = else_block.result_span();
                let else_block = self.visit_block(else_block);

                // check type equality of the `then` and `else` branches
                result_type = match (then_block.result_type, else_block.result_type) {
                    // unknown when any branch is unknown
                    (Type::Unknown, _) | (_, Type::Unknown) => Type::Unknown,
                    // never when both branches are never
                    (Type::Never, Type::Never) => Type::Never,
                    // unit when both branches are either unit or never
                    (Type::Unit | Type::Never, Type::Unit | Type::Never) => Type::Unit,
                    // the then_type when both branches have the same type
                    (then_type, else_type) if then_type == else_type => then_type,
                    // unknown and error otherwise
                    _ => {
                        self.error(
                            ErrorKind::Type,
                            format!(
                                "mismatched types: expected `{}`, found `{}`",
                                then_block.result_type, else_block.result_type
                            ),
                            vec!["the `if` and `else` branches result in the same type".into()],
                            else_result_span,
                        );
                        self.hint("expected due to this", then_result_span);
                        Type::Unknown
                    }
                };

                Some(else_block)
            }
            None => {
                result_type = match then_block.result_type {
                    Type::Unknown => Type::Unknown,
                    Type::Unit | Type::Never => Type::Unit,
                    _ => {
                        self.error(
                            ErrorKind::Type,
                            format!(
                                "mismatched types: missing else branch with `{}` result type",
                                then_block.result_type
                            ),
                            vec![format!("the `if` branch results in `{}`, therefore an else branch was expected", then_block.result_type).into()],
                            node.span,
                        );
                        Type::Unknown
                    }
                };

                None
            }
        };

        let constant = cond.constant()
            && then_block.constant
            && else_block.as_ref().map_or(false, |block| block.constant);

        AnalyzedExpression::If(
            AnalyzedIfExpr {
                result_type,
                constant,
                cond,
                then_block,
                else_block,
            }
            .into(),
        )
    }

    fn visit_ident_expr(&mut self, node: Spanned<&'src str>) -> AnalyzedExpression<'src> {
        let result_type = match self.scope_mut().vars.get_mut(node.inner) {
            Some(var) => {
                var.used = true;
                var.type_
            }
            None => {
                self.error(
                    ErrorKind::Reference,
                    format!("use of undeclared variable `{}`", node.inner),
                    vec![],
                    node.span,
                );
                Type::Unknown
            }
        };

        AnalyzedExpression::Ident(AnalyzedIdentExpr {
            result_type,
            ident: node.inner,
        })
    }

    fn visit_prefix_expr(&mut self, node: PrefixExpr<'src>) -> AnalyzedExpression<'src> {
        let expr_span = node.expr.span();
        let expr = self.visit_expression(node.expr);

        let result_type = match node.op {
            PrefixOp::Not => {
                match expr.result_type() {
                    Type::Bool => Type::Bool,
                    Type::Unknown => Type::Unknown,
                    Type::Never => {
                        self.warn_unreachable_expr(node.span, expr_span);
                        Type::Never
                    }
                    _ => {
                        self.error(
                            ErrorKind::Type,
                            format!("prefix operator `!` does not allow values of type `bool`, got `{}`", expr.result_type()),
                            vec![],
                            node.span,
                        );
                        Type::Unknown
                    }
                }
            }
            PrefixOp::Neg => match expr.result_type() {
                Type::Bool | Type::Char | Type::Unit => {
                    self.error(
                        ErrorKind::Type,
                        format!(
                            "prefix operator `-` does not allow values of type `{}`",
                            expr.result_type()
                        ),
                        vec![],
                        node.span,
                    );
                    Type::Unknown
                }
                type_ => type_,
            },
        };

        AnalyzedExpression::Prefix(
            AnalyzedPrefixExpr {
                result_type,
                constant: expr.constant(),
                op: node.op,
                expr,
            }
            .into(),
        )
    }

    fn infix_test_types(
        &mut self,
        types: &[Type],
        (left_type, right_type): (Type, Type),
        op: InfixOp,
        span: Span,
        override_result_type: Option<Type>,
        (lhs_span, rhs_span): (Span, Span),
    ) -> Type {
        match (left_type, right_type) {
            (Type::Unknown, _) | (_, Type::Unknown) => Type::Unknown,
            (Type::Never, _) => {
                self.warn_unreachable_expr(span, lhs_span);
                Type::Never
            }
            (_, Type::Never) => {
                self.warn_unreachable_expr(span, rhs_span);
                Type::Never
            }
            (left, right) if left == right && types.contains(&left) => match override_result_type {
                Some(type_) => type_,
                None => left,
            },
            (left, right) if left != right => {
                self.error(
                    ErrorKind::Type,
                    format!(
                        "infix expressions require equal types on both sides, got `{left}` and `{right}`"
                    ),
                    vec![],
                    span,
                );
                Type::Unknown
            }
            (type_, _) => {
                self.error(
                    ErrorKind::Type,
                    format!("infix operator `{op}` does not allow values of type `{type_}`"),
                    vec![],
                    span,
                );
                Type::Unknown
            }
        }
    }

    fn visit_infix_expr(&mut self, node: InfixExpr<'src>) -> AnalyzedExpression<'src> {
        let lhs_span = node.lhs.span();
        let rhs_span = node.rhs.span();
        let lhs = self.visit_expression(node.lhs);
        let rhs = self.visit_expression(node.rhs);

        let (expected_types, override_type): (&[Type], _) = match node.op {
            InfixOp::Plus | InfixOp::Minus | InfixOp::Mul | InfixOp::Div => {
                (&[Type::Int, Type::Float], None)
            }
            InfixOp::Lt | InfixOp::Gt | InfixOp::Lte | InfixOp::Gte => {
                (&[Type::Int, Type::Float], Some(Type::Bool))
            }
            InfixOp::Rem | InfixOp::Pow | InfixOp::Shl | InfixOp::Shr => (&[Type::Int], None),
            InfixOp::Eq | InfixOp::Neq => (
                &[Type::Int, Type::Float, Type::Bool, Type::Char],
                Some(Type::Bool),
            ),
            InfixOp::BitOr | InfixOp::BitAnd | InfixOp::BitXor => (&[Type::Int, Type::Bool], None),
            InfixOp::And | InfixOp::Or => (&[Type::Bool], None),
        };
        let result_type = self.infix_test_types(
            expected_types,
            (lhs.result_type(), rhs.result_type()),
            node.op,
            node.span,
            override_type,
            (lhs_span, rhs_span),
        );

        AnalyzedExpression::Infix(
            AnalyzedInfixExpr {
                result_type,
                constant: lhs.constant() && rhs.constant(),
                lhs,
                op: node.op,
                rhs,
            }
            .into(),
        )
    }

    fn assign_type_error(&mut self, op: AssignOp, type_: Type, span: Span) -> Type {
        self.error(
            ErrorKind::Type,
            format!("assignment operator `{op}` does not allow values of type `{type_}`"),
            vec![],
            span,
        );
        Type::Unknown
    }

    fn visit_assign_expr(&mut self, node: AssignExpr<'src>) -> AnalyzedExpression<'src> {
        let var_type = match self.scope().vars.get(node.assignee.inner) {
            Some(var) => var.type_,
            None => match self.functions.get(node.assignee.inner) {
                Some(_) => {
                    self.error(
                        ErrorKind::Type,
                        "cannot assign to functions",
                        vec![],
                        node.assignee.span,
                    );
                    Type::Unknown
                }
                None => {
                    self.error(
                        ErrorKind::Reference,
                        format!("use of undeclared name `{}`", node.assignee.inner),
                        vec![],
                        node.assignee.span,
                    );
                    Type::Unknown
                }
            },
        };

        let expr_span = node.expr.span();
        let expr = self.visit_expression(node.expr);
        let result_type = match (node.op, var_type, expr.result_type()) {
            (_, Type::Unknown, _) | (_, _, Type::Unknown) => Type::Unknown,
            (_, _, Type::Never) => {
                self.warn_unreachable_expr(node.span, expr_span);
                Type::Never
            }
            (_, left, right) if left != right => {
                self.error(
                    ErrorKind::Type,
                    format!("mismatched types: expected `{left}`, found `{right}`"),
                    vec![],
                    expr_span,
                );
                self.hint(
                    format!("this variable has type `{left}`"),
                    node.assignee.span,
                );
                Type::Unknown
            }
            (AssignOp::Plus | AssignOp::Minus | AssignOp::Mul | AssignOp::Div, _, type_)
                if ![Type::Int, Type::Float].contains(&type_) =>
            {
                self.assign_type_error(node.op, type_, expr_span)
            }
            (AssignOp::Rem | AssignOp::Pow | AssignOp::Shl | AssignOp::Shr, _, type_)
                if type_ != Type::Int =>
            {
                self.assign_type_error(node.op, type_, expr_span)
            }
            (AssignOp::BitOr | AssignOp::BitAnd | AssignOp::BitXor, _, type_)
                if ![Type::Int, Type::Bool].contains(&type_) =>
            {
                self.assign_type_error(node.op, type_, expr_span)
            }
            (_, _, _) => Type::Unit,
        };

        AnalyzedExpression::Assign(
            AnalyzedAssignExpr {
                result_type,
                assignee: node.assignee.inner,
                op: node.op,
                expr,
            }
            .into(),
        )
    }

    fn visit_call_expr(&mut self, node: CallExpr<'src>) -> AnalyzedExpression<'src> {
        let func = match self.functions.get_mut(node.func.inner) {
            Some(func) => {
                func.used = true;
                Some((
                    func.return_type.inner.unwrap_or(Type::Unit),
                    func.params.clone(),
                ))
            }
            // TODO: builtin functions
            None => {
                self.error(
                    ErrorKind::Reference,
                    format!("use of undeclared function `{}`", node.func.inner),
                    vec![format!(
                        "it can be declared like this: `fn {}(...) {{ ... }}`",
                        node.func.inner
                    )
                    .into()],
                    node.func.span,
                );
                None
            }
        };
        let (result_type, args) = match func {
            Some((func_type, func_params)) => {
                if node.args.len() != func_params.inner.len() {
                    self.error(
                        ErrorKind::Reference,
                        format!(
                            "function `{}` takes {} arguments, however {} were supplied",
                            node.func.inner,
                            func_params.inner.len(),
                            node.args.len()
                        ),
                        vec![],
                        node.span,
                    );
                    self.hint(
                        format!(
                            "function `{}` defined here with {} parameters",
                            node.func.inner,
                            func_params.inner.len()
                        ),
                        func_params.span,
                    );
                    (func_type, vec![])
                } else {
                    let mut result_type = func_type;
                    let args = node
                        .args
                        .into_iter()
                        .zip(func_params.inner)
                        .map(|(arg, param)| {
                            let arg_span = arg.span();
                            let arg = self.visit_expression(arg);

                            match (arg.result_type(), param.1.inner) {
                                (Type::Unknown, _) | (_, Type::Unknown) => {}
                                (Type::Never, _) => {
                                    self.warn_unreachable_expr(node.span, arg_span);
                                    result_type = Type::Never;
                                }
                                (arg_type, param_type) if arg_type != param_type => {
                                    self.error(
                                        ErrorKind::Type,
                                        format!("mismatched types: expected `{param_type}`, found `{arg_type}`"),
                                        vec![],
                                        arg_span,
                                    )
                                }
                                _ => {}
                            }

                            arg
                        })
                        .collect();
                    (result_type, args)
                }
            }
            None => {
                let mut result_type = Type::Unknown;
                let args = node
                    .args
                    .into_iter()
                    .map(|arg| {
                        let arg = self.visit_expression(arg);
                        if arg.result_type() == Type::Never {
                            result_type = Type::Never;
                        }
                        arg
                    })
                    .collect();
                (result_type, args)
            }
        };

        AnalyzedExpression::Call(
            AnalyzedCallExpr {
                result_type,
                func: node.func.inner,
                args,
            }
            .into(),
        )
    }

    fn visit_cast_expr(&mut self, node: CastExpr<'src>) -> AnalyzedExpression<'src> {
        let expr_span = node.expr.span();
        let expr = self.visit_expression(node.expr);

        let result_type = match (expr.result_type(), node.type_.inner) {
            (Type::Unknown, _) => Type::Unknown,
            (Type::Never, _) => {
                self.warn_unreachable_expr(node.span, expr_span);
                Type::Never
            }
            (
                Type::Int | Type::Float | Type::Bool | Type::Char,
                Type::Int | Type::Float | Type::Bool | Type::Char,
            ) => node.type_.inner,
            _ => {
                self.error(
                    ErrorKind::Type,
                    format!(
                        "invalid cast: cannot cast type `{}` to `{}",
                        expr.result_type(),
                        node.type_.inner
                    ),
                    vec![],
                    node.span,
                );
                Type::Unknown
            }
        };

        AnalyzedExpression::Cast(
            AnalyzedCastExpr {
                result_type,
                constant: expr.constant(),
                expr,
                type_: node.type_.inner,
            }
            .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rush_parser::{span, tree};

    fn program_test(
        parsed_tree: Program,
        analyzed_tree: AnalyzedProgram,
    ) -> Result<(), Vec<Diagnostic>> {
        let (tree, diagnostics) = dbg!(Analyzer::new().analyze(parsed_tree))?;
        assert!(!diagnostics
            .iter()
            .any(|diag| matches!(diag.level, DiagnosticLevel::Error(_))));
        assert_eq!(tree, analyzed_tree);
        Ok(())
    }

    #[test]
    fn programs() -> Result<(), Vec<Diagnostic>> {
        // fn add(left: int, right: int) -> int { return left + right; } fn main() {}
        program_test(
            tree! {
                (Program @ 0..61, [
                    (FunctionDefinition @ 0..61,
                        name: ("add", @ 3..6),
                        params @ 6..29: [
                            (("left", @ 7..11), (Type::Int, @ 13..16)),
                            (("right", @ 18..23), (Type::Int, @ 25..28))],
                        return_type: (Some(Type::Int), @ 33..36),
                        block: (Block @ 37..61,
                            stmts: [
                                (ReturnStmt @ 39..59, (Some(InfixExpr @ 46..58,
                                    lhs: (Ident "left", @ 46..50),
                                    op: InfixOp::Plus,
                                    rhs: (Ident "right", @ 53..58))))],
                            expr: (None))),
                    (FunctionDefinition @ 62..74,
                        name: ("main", @ 65..69),
                        params @ 69..71: [],
                        return_type: (None, @ 70..73),
                        block: (Block @ 72..74,
                            stmts: [],
                            expr: (None)))])
            },
            analyzed_tree! {
                (Program,
                    functions: [
                        (FunctionDefinition,
                            used: false,
                            name: "add",
                            params: [
                                ("left", Type::Int),
                                ("right", Type::Int)],
                            return_type: Type::Int,
                            block: (Block -> Type::Never,
                                constant: false,
                                stmts: [
                                    (ReturnStmt, (Some(InfixExpr -> Type::Int,
                                        constant: false,
                                        lhs: (Ident -> Type::Int, "left"),
                                        op: InfixOp::Plus,
                                        rhs: (Ident -> Type::Int, "right"))))],
                                expr: (None)))],
                    main_fn: (Block -> Type::Unit,
                        constant: true,
                        stmts: [],
                        expr: (None)))
            },
        )?;

        Ok(())
    }
}

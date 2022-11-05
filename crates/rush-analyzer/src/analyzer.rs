use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use rush_parser::{ast::*, Span};

use crate::{ast::*, Diagnostic, DiagnosticLevel, ErrorKind};

#[derive(Default)]
pub struct Analyzer<'src> {
    pub functions: HashMap<&'src str, Function<'src>>,
    pub has_main_fn: bool,
    scope: Option<Scope<'src>>,
    pub diagnostics: Vec<Diagnostic>,
}

pub struct Function<'src> {
    pub ident: Spanned<&'src str>,
    pub params: Vec<(Spanned<&'src str>, Spanned<Type>)>,
    pub return_type: Spanned<Type>,
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
    fn info(&mut self, message: impl Into<Cow<'static, str>>, span: Span) {
        self.diagnostics.push(Diagnostic::new(
            DiagnosticLevel::Info,
            message,
            vec![],
            span,
        ))
    }

    /// Adds a new diagnostic with the `Warning` level
    fn warn(&mut self, message: impl Into<Cow<'static, str>>, notes: Vec<String>, span: Span) {
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
        notes: Vec<String>,
        span: Span,
    ) {
        self.diagnostics.push(Diagnostic::new(
            DiagnosticLevel::Error(kind),
            message,
            notes,
            span,
        ))
    }

    /// Analyzes a parsed AST and returns an analyzed AST whilst emmitting diagnostics
    pub fn analyze(
        mut self,
        program: Program<'src>,
    ) -> Result<(AnalyzedProgram<'src>, Vec<Diagnostic>), Vec<Diagnostic>> {
        let mut functions = vec![];
        let mut main_fn = None;

        for func in program.functions {
            let func = self.visit_function_declaration(func);
            match func.name {
                "main" => {
                    main_fn = Some(func.block);
                    self.has_main_fn = true
                }
                _ => functions.push(func),
            }
        }

        // check if there are any unused functions
        let unused_funcs: Vec<(&'src str, Span)> = self
            .functions
            .iter()
            .filter(|func| !func.1.used)
            .map(|func| (*func.0, func.1.ident.span))
            .collect();

        for (name, ident_span) in unused_funcs {
            self.warn(
                format!("function `{}` is never called", name),
                vec![format!(
                    "remove the function declaration or name it `_{name}` to hide this warning"
                )],
                ident_span,
            )
        }

        match main_fn {
            Some(main_fn) => Ok((AnalyzedProgram { functions, main_fn }, self.diagnostics)),
            None => {
                self.error(
                    ErrorKind::Semantic,
                    "missing `main` function",
                    vec![
                        "the `main` function can be implemented like this:\n\n  fn main() {\n    ...\n  }"
                            .to_string(),
                    ],
                    program.span,
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
                        "remove the variable or call it `_{name}` to hide this warning"
                    )],
                    var.span,
                );
            }
        }
    }

    /// Unwrap the current scope
    fn scope(&self) -> &Scope<'src> {
        self.scope
            .as_ref()
            .expect("statements only exist in function bodies")
    }

    /// Unwrap the current scope mutably
    fn scope_mut(&mut self) -> &mut Scope<'src> {
        self.scope
            .as_mut()
            .expect("statements only exist in function bodies")
    }

    fn visit_function_declaration(
        &mut self,
        node: FunctionDefinition<'src>,
    ) -> AnalyzedFunctionDefinition<'src> {
        // check for duplicate function names
        if self.functions.contains_key(node.name.inner) {
            self.error(
                ErrorKind::Semantic,
                "duplicate function definition",
                vec![],
                node.name.span,
            );
        }

        // check if the function is the main function
        let is_main_fn = node.name.inner == "main";

        if is_main_fn {
            // check if the main function was already defined
            if self.has_main_fn {
                self.error(
                    ErrorKind::Semantic,
                    "duplicate `main` function definition",
                    vec!["a rush program always contains exactly 1 `main` function".to_string()],
                    node.name.span,
                );
            }

            // the main function must have no parameters
            if !node.params.is_empty() {
                self.error(
                    ErrorKind::Semantic,
                    format!(
                        "the `main` function must have 0 parameters, however {} {} defined",
                        node.params.len(),
                        if node.params.len() == 1 { "is" } else { "are" },
                    ),
                    vec!["remove the parameters: `fn main() { ... }`".to_string()],
                    node.params
                        .first()
                        .expect("this error is created by a parameter")
                        .0
                        .span
                        .start
                        .until(
                            node.params
                                .last()
                                .expect("if there is a first parameter, there is a last")
                                .1
                                .span
                                .end,
                        ),
                )
            }
            // the main function must return `()`
            if node.return_type.inner != Type::Unit {
                self.error(
                    ErrorKind::Semantic,
                    format!(
                        "the `main` function's return type must be `()` but is declared as `{}`",
                        node.return_type.inner
                    ),
                    vec!["remove the return type: `fn main() { ... }`".to_string()],
                    node.return_type.span,
                )
            }
        }

        let mut scope_vars = HashMap::new();

        // check the function parameters
        let mut params = vec![];
        let mut param_names = HashSet::new();

        // only analyze parameters if this is not the main function
        if !is_main_fn {
            for (ident, type_) in node.params {
                // check for duplicate function parameters
                if !param_names.insert(ident.inner) {
                    self.error(
                        ErrorKind::Semantic,
                        "duplicate parameter name",
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

        // set scope to new blank scope
        self.scope = Some(Scope {
            fn_name: node.name.inner,
            vars: scope_vars,
        });

        // check that the block returns a legal type
        let block_result_span = node.block.result_span();
        let block = self.visit_block(node.block);
        if block.result_type != node.return_type.inner {
            self.error(
                ErrorKind::Type,
                format!(
                    "mismatched types: expected `{}`, found `{}`",
                    node.return_type.inner, block.result_type,
                ),
                vec![],
                block_result_span,
            );
            self.hint("function return type defined here", node.return_type.span);
        }

        let params_without_spans = params
            .iter()
            .map(|(ident, type_)| (ident.inner, type_.inner))
            .collect();

        // drop the scope when finished (also checks variables)
        self.drop_scope();

        // add the function to the analyzer's function list if it is not the main function
        if !is_main_fn {
            self.functions.insert(
                node.name.inner,
                Function {
                    params,
                    return_type: node.return_type.clone(),
                    used: false,
                    ident: node.name.clone(),
                },
            );
        };

        AnalyzedFunctionDefinition {
            name: node.name.inner,
            params: params_without_spans,
            return_type: node.return_type.inner,
            block,
        }
    }

    fn visit_block(&mut self, node: Block<'src>) -> AnalyzedBlock<'src> {
        let mut stmts = vec![];

        let mut is_unreachable = false;
        let mut warned_unreachable = false;

        for statement in node.stmts {
            if is_unreachable && !warned_unreachable {
                self.warn(
                    "unreachable statement",
                    vec!["there is a statement with the `!` type above this line".to_string()],
                    statement.span(),
                );
                warned_unreachable = true;
            }
            let statement = self.visit_statement(statement);
            if statement.result_type() == Type::Never {
                is_unreachable = true;
            }
            stmts.push(statement);
        }

        // possibly mark trailing expression as unreachable
        if let (Some(expr), true, false) = (&node.expr, is_unreachable, warned_unreachable) {
            self.warn(
                "unreachable expression",
                vec!["there is a statement with the `!` type above this line".to_string()],
                expr.span(),
            );
        }

        // analyze expression
        let expr = node.expr.map(|expr| self.visit_expression(expr));

        let result_type = expr.as_ref().map_or(Type::Unit, |expr| expr.result_type());
        let constant = expr.as_ref().map_or(false, |expr| expr.constant()) && stmts.is_empty();

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
        if let Some(declared) = &node.type_ {
            if declared.inner != expr.result_type() {
                self.error(
                    ErrorKind::Type,
                    format!(
                        "mismatched types: expected `{}`, found `{}`",
                        declared.inner,
                        expr.result_type(),
                    ),
                    vec![format!(
                        "you could change this statement to look like this: `let {}: {} = ...`",
                        node.name.inner,
                        expr.result_type()
                    )],
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
            if !old.used {
                self.warn(
                    format!("unused variable `{}`", node.name.inner),
                    vec![format!(
                        "remove the variable or call it `_{}` to hide this warning",
                        node.name.inner
                    )],
                    old.span,
                );
                self.hint(
                    format!("variable `{}` shadowed here", node.name.inner),
                    node.name.span,
                );
            }
        }

        AnalyzedStatement::Let(AnalyzedLetStmt {
            mutable: node.mutable,
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

        // test if the return type is legal in the current function
        if curr_fn.return_type.inner != return_type {
            let fn_type_span = curr_fn.return_type.span;

            self.error(
                ErrorKind::Type,
                format!(
                    "mismatched types: expected `{}`, found `{}`",
                    curr_fn.return_type.inner, return_type
                ),
                vec![format!(
                    "you can change the function return type like this: `fn {} (...) -> {} {{ ...",
                    self.scope().fn_name,
                    curr_fn.return_type.inner,
                )],
                node.span,
            );
            self.hint("function return type defined here", fn_type_span)
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

        // check that the type of the cond is bool
        if cond.result_type() != Type::Bool {
            self.error(
                ErrorKind::Type,
                format!(
                    "expected value of type bool, found `{}`",
                    cond.result_type()
                ),
                vec!["a if condition must be a bool value: `if true { ... }`".to_string()],
                cond_span,
            )
        } else {
            // check that the condition is non-constant
            if cond.constant() {
                self.warn(
                    "redundant if expression: condition is constant",
                    vec!["the condition always evaluates to either `true` or `false`".to_string()],
                    cond_span,
                )
            }
        }

        // analyze then_block
        let then_result_span = node.then_block.result_span();
        let then_block = self.visit_block(node.then_block);

        let mut mismatched_types = false;

        // analyze else_block if it exists
        let else_block = match node.else_block {
            Some(else_block) => {
                let else_result_span = else_block.result_span();
                let else_block = self.visit_block(else_block);

                if then_block.result_type != else_block.result_type {
                    mismatched_types = true;
                    self.error(
                        ErrorKind::Type,
                        format!(
                            "mismatched types: expected `{}`, found `{}`",
                            then_block.result_type, else_block.result_type
                        ),
                        vec![
                            "the `if` and `else` branches must result in the same type".to_string()
                        ],
                        else_result_span,
                    );
                    self.hint("expected due to this", then_result_span);
                };
                Some(else_block)
            }
            None => {
                if then_block.result_type != Type::Unit {
                    mismatched_types = true;
                    self.error(
                        ErrorKind::Type,
                        format!(
                            "mismatched types: missing else branch with `{}` result type",
                            then_block.result_type
                        ),
                        vec![format!("the `if` branch results in `{}`, therefore an else branch was expected", then_block.result_type)],
                        node.span,
                    )
                }
                None
            }
        };

        let result_type = match mismatched_types {
            true => Type::Unknown,
            false => then_block.result_type,
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
        let (result_type, kind) = match self.scope_mut().vars.get_mut(node.inner) {
            Some(var) => {
                var.used = true;
                (var.type_, AnalyzedIdentKind::Variable)
            }
            None => match self.functions.get_mut(node.inner) {
                Some(func) => {
                    func.used = true;
                    (func.return_type.inner, AnalyzedIdentKind::Function)
                }
                None => {
                    self.error(
                        ErrorKind::Reference,
                        format!("use of undeclared name `{}`", node.inner),
                        vec![],
                        node.span,
                    );
                    (Type::Unknown, AnalyzedIdentKind::Variable)
                }
            },
        };

        AnalyzedExpression::Ident(AnalyzedIdentExpr {
            result_type,
            ident: node.inner,
            kind,
        })
    }

    fn visit_prefix_expr(&mut self, node: PrefixExpr<'src>) -> AnalyzedExpression<'src> {
        let expr = self.visit_expression(node.expr);

        let result_type = match node.op {
            PrefixOp::Not => {
                match expr.result_type() {
                    Type::Bool => Type::Bool,
                    Type::Unknown => Type::Unknown,
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
        left_type: Type,
        right_type: Type,
        op: InfixOp,
        span: Span,
    ) -> Type {
        match (left_type, right_type) {
            (Type::Unknown, _) | (_, Type::Unknown) => Type::Unknown,
            (Type::Never, _) | (_, Type::Never) => Type::Never,
            (left, right) if left == right && types.contains(&left) => left,
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
        let lhs = self.visit_expression(node.lhs);
        let rhs = self.visit_expression(node.rhs);

        let expected_types = match node.op {
            InfixOp::Plus
            | InfixOp::Minus
            | InfixOp::Mul
            | InfixOp::Div
            | InfixOp::Lt
            | InfixOp::Gt
            | InfixOp::Lte
            | InfixOp::Gte => &[Type::Int, Type::Float][..],
            InfixOp::Rem | InfixOp::Pow | InfixOp::Shl | InfixOp::Shr => &[Type::Int],
            InfixOp::Eq | InfixOp::Neq => &[Type::Int, Type::Float, Type::Bool, Type::Char],
            InfixOp::BitOr | InfixOp::BitAnd | InfixOp::BitXor => &[Type::Int, Type::Bool],
            InfixOp::And | InfixOp::Or => &[Type::Bool],
        };
        let result_type = self.infix_test_types(
            expected_types,
            lhs.result_type(),
            rhs.result_type(),
            node.op,
            node.span,
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
            (_, Type::Never, _) | (_, _, Type::Never) => Type::Never,
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
        todo!()
    }

    fn visit_cast_expr(&mut self, node: CastExpr<'src>) -> AnalyzedExpression<'src> {
        todo!()
    }
}

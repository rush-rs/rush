use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use rush_parser::{ast::*, Span};

use crate::{ast::*, Diagnostic, DiagnosticLevel, ErrorKind};

#[derive(Default)]
pub struct Analyzer<'src> {
    pub functions: HashMap<&'src str, Function<'src>>,
    scope: Option<Scope<'src>>,
    pub diagnostics: Vec<Diagnostic>,
}

pub struct Function<'src> {
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
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Hint, message, span))
    }
    /// Adds a new diagnostic with the `Info` level
    fn info(&mut self, message: impl Into<Cow<'static, str>>, span: Span) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Info, message, span))
    }
    /// Adds a new diagnostic with the `Warning` level
    fn warn(&mut self, message: impl Into<Cow<'static, str>>, span: Span) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Warning, message, span))
    }
    /// Adds a new diagnostic with the `Error` level using the specified error kind
    fn error(&mut self, kind: ErrorKind, message: impl Into<Cow<'static, str>>, span: Span) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Error(kind), message, span))
    }

    /// Analyzes a parsed AST and returns an analyzed AST whilst emmitting diagnostics
    pub fn analyze(mut self, program: Program<'src>) -> AnalyzedProgram<'src> {
        program
            .functions
            .into_iter()
            .map(|func| self.visit_function_declaration(func))
            .collect()
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
            if !var.used {
                self.warn(format!("unused variable `{}`", name), var.span);
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
                node.name.span,
            );
        }

        let mut scope_vars = HashMap::new();

        // check the function parameters
        let mut params = vec![];
        let mut param_names = HashSet::new();

        for (ident, type_) in node.params {
            // check for duplicate function parameters
            if !param_names.insert(ident.inner) {
                self.error(ErrorKind::Semantic, "duplicate parameter name", ident.span);
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
                block_result_span,
            );
            self.hint("function return type defined here", node.return_type.span);
        }

        let params_without_spans = params
            .iter()
            .map(|(ident, type_)| (ident.inner, type_.inner))
            .collect();

        // add the function to the analyzer's function list
        self.functions.insert(
            node.name.inner,
            Function {
                params,
                return_type: node.return_type.clone(),
                used: false,
            },
        );

        // drop the scope when finished (also checks variables)
        self.drop_scope();

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
                self.warn("unreachable statement", statement.span());
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
            self.warn("unreachable expression", expr.span());
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
                self.warn(format!("unused variable `{}`", node.name.inner), old.span);
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
            Expression::Int(node) => todo!(),
            Expression::Float(node) => todo!(),
            Expression::Bool(node) => todo!(),
            Expression::Ident(node) => todo!(),
            Expression::Prefix(node) => todo!(),
            Expression::Infix(node) => todo!(),
            Expression::Assign(node) => todo!(),
            Expression::Call(node) => todo!(),
            Expression::Cast(node) => todo!(),
            Expression::Grouped(node) => todo!(),
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
                cond_span,
            )
        } else {
            // check that the condition is non-constant
            if cond.constant() {
                self.warn("redundant if expression: condition is constant", cond_span)
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
}

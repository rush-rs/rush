use std::collections::{HashMap, HashSet};

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
    fn hint(&mut self, message: String, span: Span) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Hint, message, span))
    }
    /// Adds a new diagnostic with the `Info` level
    fn info(&mut self, message: String, span: Span) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Info, message, span))
    }
    /// Adds a new diagnostic with the `Warning` level
    fn warn(&mut self, message: String, span: Span) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Warning, message, span))
    }
    /// Adds a new diagnostic with the `Error` level using the specified error kind
    fn error(&mut self, kind: ErrorKind, message: String, span: Span) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Error(kind), message, span))
    }

    /// Analyzes a parsed AST and returns an analyzed AST whilst emmitting diagnostics
    pub fn analyze(mut self, program: Program<'src>) -> AnalysedProgram<'src> {
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
            if var.used {
                continue;
            }
            self.warn(format!("unused variable `{}`", name), var.span);
        }
    }

    fn visit_function_declaration(
        &mut self,
        node: FunctionDefinition<'src>,
    ) -> AnalysedFunctionDefinition<'src> {
        // check for duplicate function names
        if self.functions.contains_key(node.name.inner) {
            self.error(
                ErrorKind::Semantic,
                "duplicate function definition".to_string(),
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
                self.error(
                    ErrorKind::Semantic,
                    "duplicate parameter name".to_string(),
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

        // set scope to new blank scope
        self.scope = Some(Scope {
            fn_name: node.name.inner,
            vars: scope_vars,
        });

        // check that the block returns a legal type
        let block_result_span = node
            .block
            .stmts
            .last()
            .map_or(node.block.span, |stmt| stmt.span());
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
            self.hint(
                "function return value defined here".to_string(),
                node.return_type.span,
            );
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

        AnalysedFunctionDefinition {
            name: node.name.inner,
            params: params_without_spans,
            return_type: node.return_type.inner,
            block,
        }
    }

    fn visit_block(&mut self, node: Block<'src>) -> AnalysedBlock<'src> {
        let mut stmts = vec![];
        let mut is_unreachable = false;

        for statement in node.stmts {
            if is_unreachable {
                self.warn("unreachable statement".to_string(), statement.span());
            }
            let statement = self.visit_statement(statement);
            if statement.result_type() == Type::Never {
                is_unreachable = true;
            }
            stmts.push(statement);
        }

        // possibly mark trailing expression as unreachable
        if let (Some(expr), true) = (&node.expr, is_unreachable) {
            self.warn("unreachable expression".to_string(), expr.span());
        }

        // analyze expression
        let expr = node.expr.map(|expr| self.visit_expression(expr));

        let result_type = expr.as_ref().map_or(Type::Unit, |expr| expr.result_type());
        let constant = expr.as_ref().map_or(false, |expr| expr.constant()) && stmts.is_empty();

        AnalysedBlock {
            result_type,
            constant,
            stmts,
            expr,
        }
    }

    fn visit_statement(&mut self, node: Statement<'src>) -> AnalysedStatement<'src> {
        match node {
            Statement::Let(node) => self.visit_let_stmt(node),
            Statement::Return(node) => self.visit_return_stmt(node),
            Statement::Expr(node) => self.visit_expr_stmt(node),
        }
    }

    fn visit_let_stmt(&mut self, node: LetStmt<'src>) -> AnalysedStatement<'src> {
        // save the expression's span for later use
        let expr_span = node.expr.span();

        // analyze the right hand side first
        let expr = self.visit_expression(node.expr);

        // check if the optional type conflicts with the rhs
        if let Some(declared) = node.type_ {
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
                self.hint("expected due to this".to_string(), declared.span);
            }
        }

        // insert and do additional checks if variable is shadowed
        if let Some(old) = self.scope.as_mut().unwrap().vars.insert(
            node.name.inner,
            Variable {
                type_: expr.result_type(),
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

        AnalysedStatement::Let(AnalysedLetStmt {
            mutable: node.mutable,
            name: node.name.inner,
            expr,
        })
    }

    fn visit_return_stmt(&mut self, node: ReturnStmt<'src>) -> AnalysedStatement<'src> {
        // if there is an expression, visit it
        let expr = node.expr.map(|expr| self.visit_expression(expr));

        // create the return value based on the expr (Unit as fallback)
        let return_type = expr.as_ref().map_or(Type::Unit, |expr| expr.result_type());

        let curr_fn = self
            .functions
            .get(self.scope.as_ref().unwrap().fn_name)
            .unwrap();

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
            self.hint(
                "function return value defined here".to_string(),
                fn_type_span,
            )
        }

        AnalysedStatement::Return(expr)
    }

    fn visit_expr_stmt(&mut self, node: ExprStmt<'src>) -> AnalysedStatement<'src> {
        todo!("@RubixDev will implement from here")
    }

    fn visit_expression(&mut self, node: Expression<'src>) -> AnalysedExpression<'src> {
        todo!("@RubixDev will implement from here")
    }
}

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
    pub span: Span,
    pub params: Vec<(AnnotatedIdent<'src>, AnnotatedType<'src>)>,
    pub return_type: AnnotatedType<'src>,
}

#[derive(Debug)]
pub struct Scope<'src> {
    pub fn_name: &'src str,
    pub vars: HashMap<&'src str, Variable>,
}

#[derive(Debug)]
pub struct Variable {
    pub type_: TypeKind,
    pub span: Span,
    pub used: bool,
    pub mutable: bool,
}

impl<'src> Analyzer<'src> {
    /// Creates a new [`Analyzer`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new diagnostic with the hint level
    fn hint(&mut self, message: String, span: Span) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Hint, message, span))
    }
    /// Adds a new diagnostic with the info level
    fn info(&mut self, message: String, span: Span) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Info, message, span))
    }
    /// Adds a new diagnostic with the warning level
    fn warn(&mut self, message: String, span: Span) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Warning, message, span))
    }
    /// Adds a new diagnostic with the error level using the specified error kind
    fn error(&mut self, kind: ErrorKind, message: String, span: Span) {
        self.diagnostics
            .push(Diagnostic::new(DiagnosticLevel::Error(kind), message, span))
    }

    pub fn analyze(mut self, program: ParsedProgram<'src>) -> AnnotatedProgram<'src> {
        AnnotatedProgram {
            span: program.span,
            functions: program
                .functions
                .into_iter()
                .map(|func| self.visit_function_declaration(func))
                .collect(),
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

        // analyze its values
        for (name, var) in scope.vars {
            if var.used {
                continue;
            }
            self.warn(format!("unused variable `{}`", name), var.span);
        }
    }

    fn visit_function_declaration(
        &mut self,
        function: ParsedFunctionDefinition<'src>,
    ) -> AnnotatedFunctionDefinition<'src> {
        // check for duplicate function names
        if self.functions.contains_key(function.name.value) {
            self.error(
                ErrorKind::Semantic,
                "duplicate function definition".to_string(),
                function.name.span,
            );
        }

        let mut vars = HashMap::new();

        // check the function parameters
        let mut params = vec![];
        let mut param_names = HashSet::new();
        for (ident, type_) in &function.params {
            // check for duplicate function parameters
            if !param_names.insert(ident.value) {
                self.error(
                    ErrorKind::Semantic,
                    "duplicate parameter name".to_string(),
                    ident.span,
                );
            }
            vars.insert(
                ident.value,
                Variable {
                    type_: type_.value,
                    span: ident.span,
                    used: false,
                    // TODO: maybe allow `mut` params
                    mutable: false,
                },
            );
            params.push((
                AnnotatedIdent {
                    span: ident.span,
                    annotation: Annotation::new(type_.value, false),
                    value: ident.value,
                },
                AnnotatedType {
                    span: type_.span,
                    annotation: Annotation::new(type_.value, false),
                    value: type_.value,
                },
            ));
        }

        let return_type = AnnotatedType {
            span: function.return_type.span,
            annotation: Annotation::new(function.return_type.value, false),
            value: function.return_type.value,
        };

        // set scope to new blank scope
        self.scope = Some(Scope {
            fn_name: function.name.value,
            vars,
        });

        // check that the block returns a legal type
        let block = self.visit_block(function.block);
        if block.annotation.result_type != function.return_type.value {
            self.error(
                ErrorKind::Type,
                format!(
                    "mismatched types: expected `{}`, found `{}`",
                    function.return_type.value, block.annotation.result_type,
                ),
                block.stmts.last().map_or(block.span, |stmt| stmt.span()),
            );
            self.hint(
                "function return value defined here".to_string(),
                function.return_type.span,
            );
        }

        // add the function to the analyzer's function list
        self.functions.insert(
            function.name.value,
            Function {
                span: function.span,
                params: params.clone(),
                return_type: return_type.clone(),
            },
        );

        // drop the scope when finished (also checks variables)
        self.drop_scope();

        AnnotatedFunctionDefinition {
            span: function.span,
            annotation: Annotation::new(function.return_type.value, false),
            name: AnnotatedIdent {
                span: function.name.span,
                annotation: Annotation::new(TypeKind::Unknown, false),
                value: function.name.value,
            },
            params,
            return_type,
            block,
        }
    }

    fn visit_block(&mut self, block: ParsedBlock<'src>) -> AnnotatedBlock<'src> {
        let mut stmts = vec![];
        let mut is_unreachable = false;

        for statement in block.stmts {
            if is_unreachable {
                self.warn("unreachable statement".to_string(), statement.span());
            }
            let statement = self.visit_statement(statement);
            if statement.annotation().result_type == TypeKind::Never {
                is_unreachable = true;
            }
            stmts.push(statement);
        }

        let return_type = stmts
            .last()
            .map_or(TypeKind::Unit, |l| l.annotation().result_type);

        AnnotatedBlock {
            span: block.span,
            annotation: Annotation::new(return_type, false),
            stmts,
        }
    }

    fn visit_statement(&mut self, statement: ParsedStatement<'src>) -> AnnotatedStatement<'src> {
        match statement {
            Statement::Let(node) => self.visit_let_statement(node),
            Statement::Return(node) => self.visit_return_statement(node),
            Statement::Expr(node) => self.visit_expression_stmt(node),
        }
    }

    fn visit_let_statement(&mut self, node: ParsedLetStmt<'src>) -> AnnotatedStatement<'src> {
        // analyze the right hand side first
        let expr = self.visit_expression(node.expr);

        // check if the optional type conflicts with the rhs
        if let Some(declared) = node.type_ {
            if declared.value != expr.annotation().result_type {
                self.error(
                    ErrorKind::Type,
                    format!(
                        "mismatched types: expected `{}`, found `{}`",
                        declared.value,
                        expr.annotation().result_type,
                    ),
                    expr.span(),
                );
                self.hint("expected due to this".to_string(), declared.span);
            }
        }

        // insert and do additional checks if variable is shadowed
        if let Some(old) = self.scope.as_mut().unwrap().vars.insert(
            node.name.value,
            Variable {
                type_: expr.annotation().result_type,
                span: node.name.span,
                used: false,
                mutable: node.mutable,
            },
        ) {
            // a previous variable is shadowed by this declaration, analyze its use
            if !old.used {
                self.warn(format!("unused variable `{}`", node.name.value), old.span);
                self.hint(
                    format!("variable `{}` shadowed here", node.name.value),
                    node.name.span,
                );
            }
        }

        AnnotatedStatement::Let(AnnotatedLetStmt {
            span: node.span,
            annotation: Annotation::new(TypeKind::Unit, false),
            mutable: node.mutable,
            name: AnnotatedIdent {
                span: node.name.span,
                annotation: Annotation::new(TypeKind::Unit, false),
                value: node.name.value,
            },
            type_: None,
            expr,
        })
    }

    fn visit_return_statement(&mut self, node: ParsedReturnStmt<'src>) -> AnnotatedStatement<'src> {
        // if there is an expression, visit it
        let expr = node
            .expr
            .map_or_else(|| None, |expr| Some(self.visit_expression(expr)));

        // create the return value based on the expr (Unit as fallback)
        let return_type = expr
            .as_ref()
            .map_or(TypeKind::Unit, |expr| expr.annotation().result_type);

        let curr_fn = self
            .functions
            .get(self.scope.as_ref().unwrap().fn_name)
            .unwrap();

        // test if the return type is legal in the current function
        if curr_fn.return_type.value != return_type {
            let fn_type_span = curr_fn.return_type.span;

            self.error(
                ErrorKind::Type,
                format!(
                    "mismatched types: expected `{}`, found `{}`",
                    curr_fn.return_type.value, return_type
                ),
                node.span,
            );
            self.hint(
                "function return value defined here".to_string(),
                fn_type_span,
            )
        }

        AnnotatedStatement::Return(AnnotatedReturnStmt {
            span: node.span,
            annotation: Annotation::new(TypeKind::Unit, false),
            expr,
        })
    }

    fn visit_expression_stmt(
        &mut self,
        expression: ParsedExprStmt<'src>,
    ) -> AnnotatedStatement<'src> {
        todo!("@RubixDev will implement from here")
    }

    fn visit_expression(
        &mut self,
        expression: ParsedExpression<'src>,
    ) -> AnnotatedExpression<'src> {
        todo!("@RubixDev will implement from here")
    }
}

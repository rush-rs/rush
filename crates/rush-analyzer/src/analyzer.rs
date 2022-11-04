use std::{
    collections::{HashMap, HashSet},
    fmt::format,
};

use rush_parser::{
    ast::{
        Expression, LetStmt, ParsedBlock, ParsedExprStmt, ParsedExpression,
        ParsedFunctionDefinition, ParsedLetStmt, ParsedProgram, ParsedReturnStmt, ParsedStatement,
        ReturnStmt, Statement, TypeKind,
    },
    Span,
};

use crate::{
    ast::{
        AnnotatedBlock, AnnotatedExprStmt, AnnotatedExpression, AnnotatedFunctionDefinition,
        AnnotatedIdent, AnnotatedLetStmt, AnnotatedProgram, AnnotatedStatement, AnnotatedType,
        Annotation,
    },
    Diagnostic, DiagnosticLevel, ErrorKind,
};

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

impl<'src> Default for Analyzer<'src> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'src> Analyzer<'src> {
    /// Creates a new [`Analyzer`].
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            scope: None,
            diagnostics: vec![],
        }
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
        let mut functions = vec![];
        for function in program.functions {
            let fun = self.visit_function_declaration(function);
            functions.push(fun);
        }
        AnnotatedProgram {
            span: program.span,
            functions,
        }
    }

    fn visit_function_declaration(
        &mut self,
        function: ParsedFunctionDefinition<'src>,
    ) -> AnnotatedFunctionDefinition<'src> {
        if self.functions.get(function.name.value).is_some() {
            self.error(
                ErrorKind::Semantic,
                "duplicate function definition".to_string(),
                function.name.span,
            )
        }

        // check the function parameters
        let mut params = vec![];
        let mut param_names = HashSet::new();
        for param in &function.params {
            // check for duplicate function parameters
            if !param_names.insert(param.0.value) {
                self.error(
                    ErrorKind::Semantic,
                    "duplicate parameter name".to_string(),
                    param.0.span,
                )
            }
            params.push((
                AnnotatedIdent {
                    span: param.0.span,
                    annotation: Annotation::new(param.1.value, false),
                    value: param.0.value,
                },
                AnnotatedType {
                    span: param.1.span,
                    annotation: Annotation::new(param.1.value, false),
                    value: param.1.value,
                },
            ))
        }

        let return_type = AnnotatedType {
            span: function.return_type.span,
            annotation: Annotation::new(TypeKind::Unknown, false),
            value: function.return_type.value,
        };

        // add the functin to the analyzer's function scope
        self.functions.insert(
            function.name.value,
            Function {
                span: function.span,
                params: params.clone(),
                return_type: return_type.clone(),
            },
        );

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
            block: self.visit_block(function.block),
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
            Statement::Expr(node) => self.visit_expression(node.expr),
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

        AnnotatedStatement(AnnotatedLetStmt {
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

    fn visit_return_statement(
        &mut self,
        return_stmt: ParsedReturnStmt<'src>,
    ) -> AnnotatedStatement<'src> {
        todo!()
    }

    fn visit_expression_stmt(
        &mut self,
        expression: ParsedExprStmt<'src>,
    ) -> AnnotatedExprStmt<'src> {
        todo!("@RubixDev will implement from here")
    }

    fn visit_expression(
        &mut self,
        expression: ParsedExpression<'src>,
    ) -> AnnotatedExpression<'src> {
        todo!("@RubixDev will implement from here")
    }
}

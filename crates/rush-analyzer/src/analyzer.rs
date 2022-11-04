use std::collections::{HashMap, HashSet};

use rush_parser::{
    ast::{ParsedFunctionDefinition, ParsedProgram, ParsedType, TypeKind},
    Span,
};

use crate::{
    ast::{
        AnnotatedFunctionDefinition, AnnotatedIdent, AnnotatedProgram, AnnotatedType, Annotation,
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
    pub return_type: ParsedType<'src>,
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
            functions.push(self.visit_function_declaration(function));
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
        self.functions.insert(
            function.name.value,
            Function {
                span: function.span,
                params,
                return_type: function.return_type,
            },
        );
        todo!()
    }
}

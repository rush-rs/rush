use std::collections::HashMap;

use rush_parser::{Span, ast::{Type, ParsedProgram}};

use crate::{Diagnostic, DiagnosticLevel, ErrorKind};

pub struct Analyzer<'src> {
    pub functions: HashMap<&'src str, Function<'src>>,
    scope: Option<Scope<'src>>,
    pub diagnostics: Vec<Diagnostic>,
}

pub struct Function<'src> {
    pub span: Span,
    pub params: Vec<(&'src str, Type)>,
    pub return_type: Type,
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

    pub fn analyze(program: &ParsedProgram<'src>) ->
}

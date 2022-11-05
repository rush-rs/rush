use std::{borrow::Cow, fmt::Display};

use rush_parser::{Error, Span};

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: Cow<'static, str>,
    pub hints: Vec<String>,
    pub span: Span,
}

impl From<Error> for Diagnostic {
    fn from(err: Error) -> Self {
        Self::new(
            DiagnosticLevel::Error(ErrorKind::Syntax),
            err.message,
            err.span,
        )
    }
}

impl Diagnostic {
    pub fn new(level: DiagnosticLevel, message: impl Into<Cow<'static, str>>, span: Span) -> Self {
        Self {
            level,
            message: message.into(),
            span,
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum DiagnosticLevel {
    Hint,
    Info,
    Warning,
    Error(ErrorKind),
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum ErrorKind {
    Syntax,
    Type,
    Semantic,
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Syntax => "SyntaxError",
                Self::Type => "TypeError",
                Self::Semantic => "SemanticError",
            }
        )
    }
}

impl ErrorKind {
    pub fn into_diagnostic(self, message: impl Into<Cow<'static, str>>, span: Span) -> Diagnostic {
        Diagnostic {
            level: DiagnosticLevel::Error(self),
            message: message.into(),
            hints: vec![],
            span,
        }
    }
}

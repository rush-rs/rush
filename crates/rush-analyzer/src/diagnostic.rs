use std::borrow::Cow;

use rush_parser::{Error, Span};

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: Cow<'static, str>,
    pub span: Span,
}

impl From<Error> for Diagnostic {
    fn from(err: Error) -> Self {
        Self::from(&err)
    }
}

impl From<&Error> for Diagnostic {
    fn from(err: &Error) -> Self {
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

impl ErrorKind {
    pub fn into_diagnostic(self, message: impl Into<Cow<'static, str>>, span: Span) -> Diagnostic {
        Diagnostic {
            level: DiagnosticLevel::Error(self),
            message: message.into(),
            span,
        }
    }
}

use rush_parser::{Error, Span};

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
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
    pub fn new(level: DiagnosticLevel, message: String, span: Span) -> Self {
        Self {
            level,
            message,
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
    pub fn into_diagnostic(self, message: String, span: Span) -> Diagnostic {
        Diagnostic {
            level: DiagnosticLevel::Error(self),
            message,
            span,
        }
    }
}

use rush_parser::Span;

#[derive(PartialEq, Eq, Debug)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
    pub span: Span,
}

impl From<rush_parser::Error> for Diagnostic {
    fn from(err: rush_parser::Error) -> Self {
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

#[derive(PartialEq, Eq, Debug)]
pub enum DiagnosticLevel {
    Info,
    Warning,
    Error(ErrorKind),
}

#[derive(PartialEq, Eq, Debug)]
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
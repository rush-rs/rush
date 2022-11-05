use std::{borrow::Cow, fmt::Display};

use rush_parser::{Error, Span};

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: Cow<'static, str>,
    pub notes: Vec<String>,
    pub span: Span,
}

impl From<Error> for Diagnostic {
    fn from(err: Error) -> Self {
        Self::new(
            DiagnosticLevel::Error(ErrorKind::Syntax),
            err.message,
            vec![],
            err.span,
        )
    }
}

impl Diagnostic {
    pub fn new(
        level: DiagnosticLevel,
        message: impl Into<Cow<'static, str>>,
        notes: Vec<String>,
        span: Span,
    ) -> Self {
        Self {
            level,
            message: message.into(),
            notes,
            span,
        }
    }

    pub fn display(&self, source_code: &str, filename: &str) -> String {
        let lines = source_code.split('\n').collect::<Vec<&str>>();

        let (raw_marker, color) = match self.level {
            DiagnosticLevel::Hint => ("~", 5),
            DiagnosticLevel::Info => ("~", 6),
            DiagnosticLevel::Warning => ("~", 3),
            DiagnosticLevel::Error(_) => ("^", 1),
        };

        let notes = self
            .notes
            .iter()
            .map(|note| format!("\x1b[1;34mnote:\x1b[0m {note}"))
            .collect::<Vec<String>>()
            .join("\n");


        // take special action if the source code is empty
        if source_code.is_empty() {
            return format!(
                "\x1b[1;3{}m{}\x1b[39m in {}\x1b[0m\n{}\n{}",
                color, self.level, filename, self.message, notes
            );
        }

        let line1 = match self.span.start.line > 1 {
            true => format!(
                "\n \x1b[90m{: >3} | \x1b[0m{}",
                self.span.start.line - 1,
                lines[self.span.start.line - 2]
            ),
            false => String::new(),
        };

        let line2 = format!(
            " \x1b[90m{: >3} | \x1b[0m{}",
            self.span.start.line,
            lines[self.span.start.line - 1]
        );

        let line3 = match self.span.start.line < lines.len() {
            true => format!(
                "\n \x1b[90m{: >3} | \x1b[0m{}",
                self.span.start.line + 1,
                lines[self.span.start.line]
            ),
            false => String::new(),
        };

        let markers = match self.span.start.line == self.span.end.line {
            true => raw_marker.repeat(self.span.end.column - self.span.start.column),
            false => raw_marker.to_string(),
        };

        let marker = format!(
            "{}\x1b[1;3{}m{}\x1b[0m",
            " ".repeat(self.span.start.column + 6),
            color,
            markers
        );

        format!(
            "\x1b[1;3{}m{}\x1b[39m at {}:{}:{}\x1b[0m\n{}\n{}\n{}{}\n\n\x1b[1;3{}m{}\x1b[0m\n{}",
            color,
            self.level,
            filename,
            self.span.start.line,
            self.span.start.column,
            line1,
            line2,
            marker,
            line3,
            color,
            self.message,
            notes,
        )
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum DiagnosticLevel {
    Hint,
    Info,
    Warning,
    Error(ErrorKind),
}

impl Display for DiagnosticLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Hint => "Hint".to_string(),
                Self::Info => "Info".to_string(),
                Self::Warning => "Warning".to_string(),
                Self::Error(kind) => format!("{kind}"),
            }
        )
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum ErrorKind {
    Syntax,
    Type,
    Semantic,
    Reference,
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}Error")
    }
}

impl ErrorKind {
    pub fn into_diagnostic(self, message: impl Into<Cow<'static, str>>, span: Span) -> Diagnostic {
        Diagnostic {
            level: DiagnosticLevel::Error(self),
            message: message.into(),
            notes: vec![],
            span,
        }
    }
}

use std::{borrow::Cow, fmt::Display};

use rush_parser::{Error, Span};

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Diagnostic<'src> {
    pub level: DiagnosticLevel,
    pub message: Cow<'static, str>,
    pub notes: Vec<Cow<'static, str>>,
    pub span: Span<'src>,
}

impl<'src> From<Error<'src>> for Diagnostic<'src> {
    fn from(err: Error<'src>) -> Self {
        Self::new(
            DiagnosticLevel::Error(ErrorKind::Syntax),
            err.message,
            vec![],
            err.span,
        )
    }
}

impl<'src> Diagnostic<'src> {
    pub fn new(
        level: DiagnosticLevel,
        message: impl Into<Cow<'static, str>>,
        notes: Vec<Cow<'static, str>>,
        span: Span<'src>,
    ) -> Self {
        Self {
            level,
            message: message.into(),
            notes,
            span,
        }
    }

    pub fn display(&self, source_code: &str, filename: &str) -> String {
        let lines: Vec<_> = source_code.split('\n').collect();

        let (raw_marker, raw_marker_single, color) = match self.level {
            DiagnosticLevel::Hint => ("~", "^", 5),     // magenta
            DiagnosticLevel::Info => ("~", "^", 4),     // blue
            DiagnosticLevel::Warning => ("~", "^", 3),  // yellow
            DiagnosticLevel::Error(_) => ("^", "^", 1), // red
        };

        let notes: String = self
            .notes
            .iter()
            .map(|note| format!("\n \x1b[1;36mnote:\x1b[0m {note}"))
            .collect();

        // take special action if the source code is empty or there is no useful span
        if source_code.is_empty() || self.span == Span::dummy() {
            return format!(
                " \x1b[1;3{color}m{lvl}\x1b[39m in {filename}\x1b[0m \n{msg}{notes}",
                lvl = self.level,
                msg = self.message,
            );
        }

        let line1 = match self.span.start.line > 1 {
            true => format!(
                "\n \x1b[90m{: >3} | \x1b[0m{}",
                self.span.start.line - 1,
                lines[self.span.start.line - 2],
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

        let markers = match (
            self.span.start.line == self.span.end.line,
            self.span.start.column + 1 == self.span.end.column,
        ) {
            // same line, wide column difference
            (true, false) => raw_marker.repeat(self.span.end.column - self.span.start.column),
            // same line, just one column difference
            (true, true) => raw_marker_single.to_string(),
            // multiline span
            (_, _) => {
                format!(
                    "{} ...\n{}\x1b[1;32m+ {} more line{}\x1b[1;0m",
                    raw_marker
                        .repeat(lines[self.span.start.line - 1].len() - self.span.start.column + 1),
                    " ".repeat(self.span.start.column + 6),
                    self.span.end.line - self.span.start.line,
                    if self.span.end.line - self.span.start.line == 1 {
                        ""
                    } else {
                        "s"
                    },
                )
            }
        };

        let marker = format!(
            "{space}\x1b[1;3{color}m{markers}\x1b[0m",
            space = " ".repeat(self.span.start.column + 6),
        );

        format!(
            " \x1b[1;3{color}m{lvl}\x1b[39m at {filename}:{line}:{col}\x1b[0m\n{line1}\n{line2}\n{marker}{line3}\n\n \x1b[1;3{color}m{msg}\x1b[0m{notes}",
            lvl = self.level,
            line = self.span.start.line,
            col = self.span.start.column,
            msg = self.message,
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
        match self {
            Self::Hint | Self::Info | Self::Warning => write!(f, "{self:?}"),
            Self::Error(kind) => write!(f, "{kind}"),
        }
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

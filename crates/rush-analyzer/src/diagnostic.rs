use std::{
    borrow::Cow,
    fmt::{self, Display, Formatter},
};

use rush_parser::{Error, Span};

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Diagnostic<'src> {
    pub level: DiagnosticLevel,
    pub message: Cow<'static, str>,
    pub notes: Vec<Cow<'static, str>>,
    pub span: Span<'src>,
    pub source: &'src str,
}

impl<'src> From<Error<'src>> for Diagnostic<'src> {
    fn from(err: Error<'src>) -> Self {
        Self::new(
            DiagnosticLevel::Error(ErrorKind::Syntax),
            err.message,
            vec![],
            err.span,
            err.source,
        )
    }
}

impl<'src> From<Box<Error<'src>>> for Diagnostic<'src> {
    fn from(err: Box<Error<'src>>) -> Self {
        Self::from(*err)
    }
}

impl<'src> Diagnostic<'src> {
    pub fn new(
        level: DiagnosticLevel,
        message: impl Into<Cow<'static, str>>,
        notes: Vec<Cow<'static, str>>,
        span: Span<'src>,
        source: &'src str,
    ) -> Self {
        Self {
            level,
            message: message.into(),
            notes,
            span,
            source,
        }
    }
}

impl Display for Diagnostic<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let ansi_col = |col: u8, bold: bool| -> Cow<'static, str> {
            match (f.alternate(), bold) {
                (true, true) => format!("\x1b[1;{col}m").into(),
                (true, false) => format!("\x1b[{col}m").into(),
                (false, _) => "".into(),
            }
        };
        let ansi_reset = match f.alternate() {
            true => "\x1b[0m",
            false => "",
        };

        let lines: Vec<_> = self.source.split('\n').collect();

        let (raw_marker, raw_marker_single, color) = match self.level {
            DiagnosticLevel::Hint => ("~", "^", 5),     // magenta
            DiagnosticLevel::Info => ("~", "^", 4),     // blue
            DiagnosticLevel::Warning => ("~", "^", 3),  // yellow
            DiagnosticLevel::Error(_) => ("^", "^", 1), // red
        };

        let notes: String = self
            .notes
            .iter()
            .map(|note| {
                format!(
                    "\n {color}note:{ansi_reset} {note}",
                    color = ansi_col(36, true),
                )
            })
            .collect();

        // take special action if the source code is empty or there is no useful span
        if self.source.is_empty() || self.span == Span::dummy() {
            return write!(
                f,
                " {color}{lvl}{reset_col} in {path}{ansi_reset} \n{msg}{notes}",
                color = ansi_col(color + 30, true),
                lvl = self.level,
                reset_col = ansi_col(39, false),
                path = self.span.start.path,
                msg = self.message,
            );
        }

        let line1 = match self.span.start.line > 1 {
            true => format!(
                "\n {}{: >3} | {ansi_reset}{}",
                ansi_col(90, false),
                self.span.start.line - 1,
                lines[self.span.start.line - 2],
            ),
            false => String::new(),
        };

        let line2 = format!(
            " {}{: >3} | {ansi_reset}{}",
            ansi_col(90, false),
            self.span.start.line,
            lines[self.span.start.line - 1]
        );

        let line3 = match self.span.start.line < lines.len() {
            true => format!(
                "\n {}{: >3} | {ansi_reset}{}",
                ansi_col(90, false),
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
                    "{marker} ...\n{space}{color}+ {line_count} more line{s}{ansi_reset}",
                    marker = raw_marker
                        .repeat(lines[self.span.start.line - 1].len() - self.span.start.column + 1),
                    space = " ".repeat(self.span.start.column + 6),
                    color = ansi_col(32, true),
                    line_count = self.span.end.line - self.span.start.line,
                    s = match self.span.end.line - self.span.start.line == 1 {
                        true => "",
                        false => "s",
                    },
                )
            }
        };

        let marker = format!(
            "{space}{color}{markers}{ansi_reset}",
            color = ansi_col(color + 30, true),
            space = " ".repeat(self.span.start.column + 6),
        );

        write!(
            f,
            " {color}{lvl}{reset_col} at {path}:{line}:{col}{ansi_reset}\n{line1}\n{line2}\n{marker}{line3}\n\n {color}{msg}{ansi_reset}{notes}",
            color = ansi_col(color + 30, true),
            lvl = self.level,
            reset_col = ansi_col(39, false),
            path = self.span.start.path,
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

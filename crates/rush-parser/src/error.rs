use crate::Span;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub span: Span,
}

impl Error {
    pub fn new(kind: ErrorKind, message: String, span: Span) -> Self {
        Self {
            kind,
            message,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Syntax,
}

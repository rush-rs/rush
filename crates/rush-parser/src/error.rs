use crate::Span;

pub type Result<'src, T> = std::result::Result<T, Box<Error<'src>>>;

#[derive(Debug, PartialEq, Eq)]
pub struct Error<'src> {
    pub message: String,
    pub span: Span<'src>,
    pub source: &'src str,
}

impl<'src> Error<'src> {
    pub fn new(message: String, span: Span<'src>, source: &'src str) -> Self {
        Self {
            message,
            span,
            source,
        }
    }

    pub fn new_boxed(message: String, span: Span<'src>, source: &'src str) -> Box<Self> {
        Self::new(message, span, source).into()
    }
}

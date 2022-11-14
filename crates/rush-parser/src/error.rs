use crate::Span;

pub type Result<'src, T> = std::result::Result<T, Error<'src>>;

#[derive(Debug, PartialEq, Eq)]
pub struct Error<'src> {
    pub message: String,
    pub span: Span<'src>,
}

impl<'src> Error<'src> {
    pub fn new(message: String, span: Span<'src>) -> Self {
        Self { message, span }
    }
}

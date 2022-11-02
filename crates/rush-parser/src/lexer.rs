use crate::error::Result;
use crate::Token;

pub struct Lexer<'src> {
    input: &'src str,
}

impl<'src> Lexer<'src> {
    pub fn new(text: &'src str) -> Self {
        todo!()
    }

    pub fn next_token(&mut self) -> Result<Token<'src>> {
        todo!()
    }
}

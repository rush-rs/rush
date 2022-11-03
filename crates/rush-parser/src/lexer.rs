use std::mem;
use std::str::Chars;

use crate::error::Result;
use crate::{Error, ErrorKind, Location, Span, Token, TokenKind};

pub struct Lexer<'src> {
    input: &'src str,
    reader: Chars<'src>,
    location: Location,
    curr_char: Option<char>,
    next_char: Option<char>,
}

macro_rules! char_construct {
    ($self:ident, $kind_single:ident, $kind_with_eq:tt, $kind_double:tt, $kind_double_with_eq:tt $(,)?) => {
        return Ok($self.make_char_construct(
            TokenKind::$kind_single,
            char_construct!(@optional $kind_with_eq),
            char_construct!(@optional $kind_double),
            char_construct!(@optional $kind_double_with_eq),
        ))
    };
    (@optional _) => { None };
    (@optional $kind:ident) => { Some(TokenKind::$kind) };
}

impl<'src> Lexer<'src> {
    pub fn new(text: &'src str) -> Self {
        let mut lexer = Self {
            input: text,
            reader: text.chars(),
            location: Location::default(),
            curr_char: None,
            next_char: None,
        };
        // Advance the lexer 2 times so that curr_char and next_char are populated
        lexer.next();
        lexer.next();
        lexer
    }

    fn next(&mut self) {
        if let Some(current_char) = self.curr_char {
            self.location.advance(
                current_char == '\n',
                // Byte offset is specified because advance does not know about the current char
                current_char.len_utf8(),
            );
        }
        // Swap the current and next char so that the old next is the new current
        mem::swap(&mut self.curr_char, &mut self.next_char);
        self.next_char = self.reader.next()
    }

    pub fn next_token(&mut self) -> Result<Token<'src>> {
        while let Some(current) = self.curr_char {
            match current {
                ' ' | '\t' | '\n' | '\r' => self.next(),
                '/' => self.skip_comment(),
                _ => break,
            }
        }

        let start_loc = self.location;
        let kind = match self.curr_char {
            None => TokenKind::Eof,
            Some('\'') => return self.make_char(),
            Some('/') => TokenKind::Slash,
            Some('(') => TokenKind::LParen,
            Some(')') => TokenKind::RParen,
            Some('{') => TokenKind::LBrace,
            Some('}') => TokenKind::RBrace,
            Some(',') => TokenKind::Comma,
            Some(':') => TokenKind::Colon,
            Some(';') => TokenKind::Semicolon,
            Some('!') => char_construct!(self, Not, Neq, _, _,),
            Some('-') if self.next_char == Some('>') => {
                self.next();
                self.next();
                return Ok(TokenKind::Arrow.spanned(Span::new(start_loc, self.location)));
            }
            Some('-') => char_construct!(self, Minus, MulAssign, _, _),
            Some('+') => char_construct!(self, Plus, PlusAssign, _, _),
            Some('*') => char_construct!(self, Star, MulAssign, Pow, PowAssign),
            Some('%') => char_construct!(self, Percent, RemAssign, _, _),
            Some('=') => char_construct!(self, Assign, Eq, _, _),
            Some('<') => char_construct!(self, Lt, Lte, Shl, ShlAssign),
            Some('>') => char_construct!(self, Gt, Gte, Shr, ShrAssign),
            Some('|') => char_construct!(self, BitOr, BitOrAssign, Or, _),
            Some('&') => char_construct!(self, BitAnd, BitAndAssign, And, _),
            Some('^') => char_construct!(self, BitXor, BitXorAssign, _, _),
            Some(other) => {
                if other.is_ascii_digit() {
                    return self.make_number();
                }
                if other.is_ascii_alphabetic() {
                    return Ok(self.make_name());
                }
                self.next();
                return Err(Error::new(
                    ErrorKind::Syntax,
                    format!("illegal character '{other}'"),
                    Span::new(start_loc, self.location),
                ));
            }
        };
        Ok(Token::new(kind, Span::new(start_loc, self.location)))
    }

    fn skip_comment(&mut self) {
        match self.next_char {
            Some('/') => {
                self.next();
                self.next();
                while self.curr_char != None && self.curr_char != Some('\n') {
                    self.next()
                }
                self.next();
            }
            Some('*') => {
                self.next();
                self.next();
                while let Some(current) = self.curr_char {
                    match current {
                        '*' if self.next_char == Some('/') => {
                            self.next();
                            self.next();
                            break;
                        }
                        _ => self.next(),
                    }
                }
            }
            _ => (),
        }
    }

    fn make_char_construct(
        &mut self,
        kind_single: TokenKind<'src>,
        kind_with_eq: Option<TokenKind<'src>>,
        kind_double: Option<TokenKind<'src>>,
        kind_double_with_eq: Option<TokenKind<'src>>,
    ) -> Token<'src> {
        let start_location = self.location;
        let char = self
            .curr_char
            .expect("this should only be called when self.curr_char is Some(_)");
        self.next();
        match (
            kind_with_eq,
            &kind_double,
            &kind_double_with_eq,
            self.curr_char,
        ) {
            (Some(kind), .., Some('=')) => {
                self.next();
                Token::new(kind, Span::new(start_location, self.location))
            }
            (_, Some(_), _, Some(current_char)) | (_, _, Some(_), Some(current_char))
                if current_char == char =>
            {
                self.next();
                match (kind_double, kind_double_with_eq, self.curr_char) {
                    (_, Some(kind), Some('=')) => {
                        self.next();
                        Token::new(kind, Span::new(start_location, self.location))
                    }
                    (Some(kind), ..) => Token::new(kind, Span::new(start_location, self.location)),
                    // can panic when all this is true:
                    // - `kind_double` is `None`
                    // - `kind_double_with_eq` is `Some(_)`
                    // - `self.curr_char` is not `Some('=')`
                    // however, this function is never called in that context
                    _ => unreachable!(),
                }
            }
            _ => Token::new(kind_single, Span::new(start_location, self.location)),
        }
    }

    fn make_slash(&mut self) -> Option<Token<'src>> {
        todo!()
    }

    fn make_char(&mut self) -> Result<Token<'src>> {
        self.next();
        let start_loc = self.location;

        let char = match self.curr_char {
            None => {
                self.next();
                return Err(Error::new(
                    ErrorKind::Syntax,
                    "expected ASCII character, found EOF".to_string(),
                    Span::new(start_loc, self.location),
                ));
            }
            Some('\\') => match self.next_char {
                Some('\\') => '\\' as u8,
                Some('\'') => '\'' as u8,
                Some('b') => '\x08' as u8,
                Some('n') => '\n' as u8,
                Some('r') => '\r' as u8,
                Some('t') => '\t' as u8,
                Some('x') => {
                    self.next();
                    let start_hex = self.location.byte_idx;
                    for _ in 0..2 {
                        if self.curr_char.is_some()
                            && self
                                .curr_char
                                .expect("This was checked beforehand")
                                .is_ascii_hexdigit()
                        {
                            self.next();
                        };
                    }
                    u8::from_str_radix(&self.input[start_hex..self.location.byte_idx], 16)
                        .expect("This string slice should be valid hexadecimal")
                }
                Some(other) => {
                    self.next();
                    return Err(Error::new(
                        ErrorKind::Syntax,
                        format!("expected excape character, found {other}"),
                        Span::new(start_loc, self.location),
                    ));
                }
                None => {
                    self.next();
                    return Err(Error::new(
                        ErrorKind::Syntax,
                        "expected escape character, found EOF".to_string(),
                        Span::new(start_loc, self.location),
                    ));
                }
            },
            Some(other) if other.is_ascii() => other as u8,
            _ => unreachable!(),
        };
        self.next();
        Ok(Token::new(
            TokenKind::Char(char),
            Span::new(start_loc, self.location),
        ))
    }

    fn make_number(&mut self) -> Result<Token<'src>> {
        let start_loc = self.location;

        while self
            .curr_char
            .map_or(false, |current| current.is_ascii_digit())
        {
            self.next();
        }

        match self.curr_char {
            Some('.') => {
                self.next();

                if self
                    .curr_char
                    .map_or(false, |current| !current.is_ascii_digit())
                {
                    let err_start = self.location;
                    self.next();
                    return Err(Error::new(
                        ErrorKind::Syntax,
                        format!(
                            "expected digit, found '{}'",
                            self.curr_char.map_or("EOF".to_string(), |c| c.to_string())
                        ),
                        Span::new(err_start, self.location),
                    ));
                }

                while self
                    .curr_char
                    .map_or(false, |current| current.is_ascii_digit())
                {
                    self.next();
                }

                let float = &self.input[start_loc.byte_idx..self.location.byte_idx]
                    .replace('_', "")
                    .parse::<f64>()
                    .expect("The grammar guarantees correctly formed float literals");
                return Ok(Token::new(
                    TokenKind::Float(*float),
                    Span::new(start_loc, self.location),
                ));
            }
            Some('f') => {
                let float = &self.input[start_loc.byte_idx..self.location.byte_idx]
                    .replace('_', "")
                    .parse::<f64>()
                    .expect("The grammar guarantees correctly formed float literals");
                return Ok(Token::new(
                    TokenKind::Float(*float),
                    Span::new(start_loc, self.location),
                ));
            }
            _ => {
                let int = match self.input[start_loc.byte_idx..self.location.byte_idx]
                    .replace('_', "")
                    .parse::<i64>()
                {
                    Ok(value) => value,
                    Err(err) => {
                        return Err(Error::new(
                            ErrorKind::Syntax,
                            format!("invalid decimal: {err}"),
                            Span::new(start_loc, self.location),
                        ))
                    }
                };
                return Ok(Token::new(
                    TokenKind::Int(int),
                    Span::new(start_loc, self.location),
                ));
            }
        }
    }
    fn make_name(&mut self) -> Token<'src> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_numbers_with_comments() {
        let code = "
            // skipped number: 42

            /* inline comment */ 1.123
            12 /* inline comment */
            13.1 // line comment
            /* prefix */ 14.12 /* suffix */
        ";
        let mut lexer = Lexer::new(code);
        let mut tokens = vec![];
        loop {
            let current = lexer.next_token().unwrap();
            if current.kind == TokenKind::Eof {
                break;
            }
            tokens.push(current);
        }
        let expected_kinds = vec![
            TokenKind::Float(1.123),
            TokenKind::Int(12),
            TokenKind::Float(13.1),
            TokenKind::Float(14.12),
        ];
        for (expected, actual) in expected_kinds.iter().zip(&tokens) {
            assert_eq!(*expected, actual.kind);
        }
    }

    #[test]
    fn test_lexer() {
        let code = r#"
            '\\'    // correctly escaped char literal
            '\'     // unterminated char literal
            'a'     // normal ASCII character
            '*'     // normal ASCII character
            '\b'    // escape char
            '\n'    // escape char
            '\r'    // escape char
            '\t'    // escape char
            '\x1b'  // hex escape
            '\a'    // invalid escape
            '\x1b1' // invalid hex
        "#;
        let mut lexer = Lexer::new(code);
        let tests = vec![
            (TokenKind::Char(b'\\'), None),
            (TokenKind::Char(b'_'), Some("unterminated char literal")),
            (TokenKind::Char(b'a'), None),
            (TokenKind::Char(b'*'), None),
            (TokenKind::Char(b'\x08'), None),
            (TokenKind::Char(b'\n'), None),
            (TokenKind::Char(b'\r'), None),
            (TokenKind::Char(b'\t'), None),
            (TokenKind::Char(b'\x1b'), None),
            (
                TokenKind::Char(b'_'),
                Some("expected escape character, found a"),
            ),
            (TokenKind::Char(b'_'), None),
        ];
        println!();
        for test in tests {
            let res = lexer.next_token();
            match (res, test.1, test.0) {
                (Ok(_), Some(expected), _) => panic!("Expected error: {:?}, got none", expected),
                (Err(err), None, _) => panic!("Unexpected error: {:?}", err),
                (Err(got), Some(expected), _) => assert_eq!(expected, got.message),
                (Ok(got), _, expected) => {
                    match got.kind {
                        TokenKind::Char(ch) => {
                            println!("found char: {} ({ch})", char::from_u32(ch as u32).unwrap())
                        }
                        _ => println!("{:?}", got),
                    }
                    assert_eq!(expected, got.kind)
                }
            }
        }
    }

    /*
    println!(
        "\n{}",
        tokens
            .iter()
            .map(|token| format!("{:?}", token))
            .collect::<Vec<String>>()
            .join("\n")
    );
    */
}

use std::mem;
use std::str::Chars;

use crate::error::Result;
use crate::{Error, ErrorKind, Location, Span, Token, TokenKind};

pub trait Lex<'src> {
    fn next_token(&mut self) -> Result<Token<'src>>;
}

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

impl<'src> Lex<'src> for Lexer<'src> {
    fn next_token(&mut self) -> Result<Token<'src>> {
        loop {
            match self.curr_char {
                Some(' ' | '\t' | '\n' | '\r') => self.next(),
                Some('/') if self.next_char == Some('/') || self.next_char == Some('*') => {
                    self.skip_comment()
                }
                _ => break,
            }
        }
        let start_loc = self.location;
        let kind = match self.curr_char {
            None => TokenKind::Eof,
            Some('\'') => return self.make_char(),
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
                TokenKind::Arrow
            }
            Some('-') => char_construct!(self, Minus, MinusAssign, _, _),
            Some('+') => char_construct!(self, Plus, PlusAssign, _, _),
            Some('*') => char_construct!(self, Star, MulAssign, Pow, PowAssign),
            Some('/') => char_construct!(self, Slash, DivAssign, _, _),
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
                if other.is_ascii_alphabetic() || other == '_' {
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
        self.next();
        Ok(Token::new(kind, Span::new(start_loc, self.location)))
    }
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

    fn skip_comment(&mut self) {
        match self.curr_char {
            Some('/') => {
                self.next();
                while self.curr_char != None && self.curr_char != Some('\n') {
                    self.next()
                }
                self.next();
            }
            Some('*') => {
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

    fn make_char(&mut self) -> Result<Token<'src>> {
        let start_loc = self.location;
        self.next();

        let char = match self.curr_char {
            None => {
                self.next();
                return Err(Error::new(
                    ErrorKind::Syntax,
                    "expected ASCII character, found EOF".to_string(),
                    Span::new(start_loc, self.location),
                ));
            }
            Some('\\') => {
                let char = match self.next_char {
                    Some('\\') => b'\\',
                    Some('\'') => b'\'',
                    Some('b') => b'\x08',
                    Some('n') => b'\n',
                    Some('r') => b'\r',
                    Some('t') => b'\t',
                    Some('x') => {
                        self.next();
                        self.next();
                        let start_hex = self.location.byte_idx;
                        let mut hex_digit_count = 0;
                        while self.curr_char.map_or(false, |c| c.is_ascii_hexdigit()) {
                            self.next();
                            hex_digit_count += 1;
                        }
                        if hex_digit_count != 2 {
                            return Err(Error::new(
                                ErrorKind::Syntax,
                                format!("expected exactly 2 hex digits, found {hex_digit_count}"),
                                Span::new(start_loc, self.location),
                            ));
                        }
                        return if self.curr_char != Some('\'') {
                            Err(Error::new(
                                ErrorKind::Syntax,
                                "unterminated char literal".to_string(),
                                Span::new(start_loc, self.location),
                            ))
                        } else {
                            let char = u8::from_str_radix(
                                &self.input[start_hex..self.location.byte_idx],
                                16,
                            )
                            .expect("This string slice should be valid hexadecimal");
                            self.next();
                            Ok(Token::new(
                                TokenKind::Char(char),
                                Span::new(start_loc, self.location),
                            ))
                        };
                    }
                    _ => {
                        self.next();
                        return Err(Error::new(
                            ErrorKind::Syntax,
                            format!(
                                "expected escape character, found {}",
                                self.curr_char.map_or("EOF".to_string(), |c| c.to_string())
                            ),
                            Span::new(start_loc, self.location),
                        ));
                    }
                };
                self.next();
                char
            }
            Some(other) if other.is_ascii() => other as u8,
            _ => unreachable!(),
        };
        self.next();
        match self.curr_char {
            Some('\'') => {
                self.next();
                Ok(Token::new(
                    TokenKind::Char(char),
                    Span::new(start_loc, self.location),
                ))
            }
            _ => {
                self.next();
                Err(Error::new(
                    ErrorKind::Syntax,
                    "unterminated char literal".to_string(),
                    Span::new(start_loc, self.location),
                ))
            }
        }
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
        let start_loc = self.location;
        while self.curr_char.map_or(false, |current| {
            current.is_ascii_alphabetic() || current.is_ascii_digit() || current == '_'
        }) {
            self.next()
        }
        self.next();
        let kind = match &self.input[start_loc.byte_idx..self.location.byte_idx] {
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "fn" => TokenKind::Fn,
            "let" => TokenKind::Let,
            "mut" => TokenKind::Mut,
            "return" => TokenKind::Return,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "as" => TokenKind::As,
            ident => TokenKind::Ident(ident),
        };
        Token::new(kind, Span::new(start_loc, self.location))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer() {
        let tests = [
            // Char tests
            ("'a'", Ok(TokenKind::Char(b'a'))),
            ("'*'", Ok(TokenKind::Char(b'*'))),
            ("'_'", Ok(TokenKind::Char(b'_'))),
            (r#"'\'"#, Err("unterminated char literal")),
            (r#"'\\'"#, Ok(TokenKind::Char(b'\\'))),
            (r#"'\a'"#, Err("expected escape character, found a")),
            (r#"'\x1b'"#, Ok(TokenKind::Char(b'\x1b'))),
            (r#"'\x1b1'"#, Err("expected exactly 2 hex digits, found 3")),
            // Keyword
            ("true", Ok(TokenKind::True)),
            ("false", Ok(TokenKind::False)),
            ("fn", Ok(TokenKind::Fn)),
            ("let", Ok(TokenKind::Let)),
            ("mut", Ok(TokenKind::Mut)),
            ("return", Ok(TokenKind::Return)),
            ("if", Ok(TokenKind::If)),
            ("else", Ok(TokenKind::Else)),
            ("as", Ok(TokenKind::As)),
            // Identifiers
            ("foo", Ok(TokenKind::Ident("foo"))),
            ("_foo", Ok(TokenKind::Ident("_foo"))),
            ("f_0o", Ok(TokenKind::Ident("f_0o"))),
            // Numbers
            ("1", Ok(TokenKind::Int(1))),
            ("42", Ok(TokenKind::Int(42))),
            ("3.1", Ok(TokenKind::Float(3.1))),
            ("42.12345678", Ok(TokenKind::Float(42.12345678))),
            ("42.69", Ok(TokenKind::Float(42.69))),
            // Paranthesis
            ("(", Ok(TokenKind::LParen)),
            (")", Ok(TokenKind::RParen)),
            ("{", Ok(TokenKind::LBrace)),
            ("}", Ok(TokenKind::RBrace)),
            // Punctuation and delimiters
            ("->", Ok(TokenKind::Arrow)),
            (",", Ok(TokenKind::Comma)),
            (":", Ok(TokenKind::Colon)),
            (";", Ok(TokenKind::Semicolon)),
            // Operators
            ("!", Ok(TokenKind::Not)),
            ("-", Ok(TokenKind::Minus)),
            ("+", Ok(TokenKind::Plus)),
            ("*", Ok(TokenKind::Star)),
            ("/", Ok(TokenKind::Slash)),
            ("%", Ok(TokenKind::Percent)),
            ("**", Ok(TokenKind::Pow)),
            ("==", Ok(TokenKind::Eq)),
            ("!=", Ok(TokenKind::Neq)),
            ("<", Ok(TokenKind::Lt)),
            (">", Ok(TokenKind::Gt)),
            ("<=", Ok(TokenKind::Lte)),
            (">=", Ok(TokenKind::Gte)),
            ("<<", Ok(TokenKind::Shl)),
            (">>", Ok(TokenKind::Shr)),
            ("|", Ok(TokenKind::BitOr)),
            ("&", Ok(TokenKind::BitAnd)),
            ("^", Ok(TokenKind::BitXor)),
            ("&&", Ok(TokenKind::And)),
            ("||", Ok(TokenKind::Or)),
            // Assignments
            ("=", Ok(TokenKind::Assign)),
            ("+=", Ok(TokenKind::PlusAssign)),
            ("-=", Ok(TokenKind::MinusAssign)),
            ("*=", Ok(TokenKind::MulAssign)),
            ("/=", Ok(TokenKind::DivAssign)),
            ("%=", Ok(TokenKind::RemAssign)),
            ("**=", Ok(TokenKind::PowAssign)),
            ("<<=", Ok(TokenKind::ShlAssign)),
            (">>=", Ok(TokenKind::ShrAssign)),
            ("|=", Ok(TokenKind::BitOrAssign)),
            ("&=", Ok(TokenKind::BitAndAssign)),
            ("^=", Ok(TokenKind::BitXorAssign)),
        ];
        println!();
        for test in tests {
            let mut lexer = Lexer::new(test.0);
            let res = lexer.next_token();
            match (res, test.1) {
                (Ok(_), Err(expected)) => panic!("Expected error: {:?}, got none", expected),
                (Err(err), Ok(_)) => panic!("Unexpected error: {:?}", err),
                (Err(got), Err(expected)) => assert_eq!(expected, got.message),
                (Ok(got), Ok(expected)) => {
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
}

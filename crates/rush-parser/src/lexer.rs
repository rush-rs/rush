use std::{mem, str::Chars};

use crate::{Error, Location, Result, Token, TokenKind};

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
        // skip comments, whitespaces and newlines
        loop {
            match (self.curr_char, self.next_char) {
                (Some(' ' | '\t' | '\n' | '\r'), _) => self.next(),
                (Some('/'), Some('/')) => self.skip_line_comment(),
                (Some('/'), Some('*')) => self.skip_block_comment(),
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
            Some('!') => char_construct!(self, Not, Neq, _, _),
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
            Some(char) if char.is_ascii_digit() => return self.make_number(),
            Some(char) if char.is_ascii_alphabetic() || char == '_' => return Ok(self.make_name()),
            Some(char) => {
                self.next();
                return Err(Error::new(
                    format!("illegal character `{char}`"),
                    start_loc.until(self.location),
                ));
            }
        };
        self.next();
        Ok(kind.spanned(start_loc.until(self.location)))
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
        // advance the lexer twice so that curr_char and next_char are populated
        lexer.next();
        lexer.next();
        lexer
    }

    fn next(&mut self) {
        if let Some(current_char) = self.curr_char {
            self.location.advance(
                current_char == '\n',
                // byte count is specified because advance does not know about the current char
                current_char.len_utf8(),
            );
        }
        // swap the current and next char so that the old next is the new current
        mem::swap(&mut self.curr_char, &mut self.next_char);
        self.next_char = self.reader.next()
    }

    fn skip_line_comment(&mut self) {
        self.next();
        self.next();
        while !matches!(self.curr_char, Some('\n') | None) {
            self.next()
        }
        self.next();
    }

    fn skip_block_comment(&mut self) {
        self.next();
        self.next();
        loop {
            match (self.curr_char, self.next_char) {
                // end of block comment
                (Some('*'), Some('/')) => {
                    self.next();
                    self.next();
                    break;
                }
                // any char in comment
                (Some(_), _) => self.next(),
                // end of file
                _ => break,
            }
        }
    }

    fn make_char_construct(
        &mut self,
        kind_single: TokenKind<'src>,
        kind_with_eq: Option<TokenKind<'src>>,
        kind_double: Option<TokenKind<'src>>,
        kind_double_with_eq: Option<TokenKind<'src>>,
    ) -> Token<'src> {
        let start_loc = self.location;
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
                kind.spanned(start_loc.until(self.location))
            }
            (_, Some(_), _, Some(current_char)) | (_, _, Some(_), Some(current_char))
                if current_char == char =>
            {
                self.next();
                match (kind_double, kind_double_with_eq, self.curr_char) {
                    (_, Some(kind), Some('=')) => {
                        self.next();
                        kind.spanned(start_loc.until(self.location))
                    }
                    (Some(kind), ..) => kind.spanned(start_loc.until(self.location)),
                    // can panic when all this is true:
                    // - `kind_double` is `None`
                    // - `kind_double_with_eq` is `Some(_)`
                    // - `self.curr_char` is not `Some('=')`
                    // however, this function is never called in that context
                    _ => unreachable!(),
                }
            }
            _ => kind_single.spanned(start_loc.until(self.location)),
        }
    }

    fn make_char(&mut self) -> Result<Token<'src>> {
        let start_loc = self.location;
        self.next();

        let char = match self.curr_char {
            None => {
                self.next();
                return Err(Error::new(
                    "unterminated char literal".to_string(),
                    start_loc.until(self.location),
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
                        for i in 0..2 {
                            if !self.curr_char.map_or(false, |c| c.is_ascii_hexdigit()) {
                                return Err(Error::new(
                                    format!("expected 2 hexadecimal digits, found {i}"),
                                    start_loc.until(self.location),
                                ));
                            }
                            self.next();
                        }
                        return match self.curr_char {
                            Some('\'') => {
                                let char = u8::from_str_radix(
                                    &self.input[start_hex..self.location.byte_idx],
                                    16,
                                )
                                .expect("This string slice should be valid hexadecimal");
                                self.next();
                                Ok(Token::new(
                                    TokenKind::Char(char),
                                    start_loc.until(self.location),
                                ))
                            }
                            _ => {
                                self.next();
                                Err(Error::new(
                                    "unterminated char literal".to_string(),
                                    start_loc.until(self.location),
                                ))
                            }
                        };
                    }
                    _ => {
                        self.next();
                        return Err(Error::new(
                            format!(
                                "expected escape character, found {}",
                                self.curr_char.map_or("EOF".to_string(), |c| c.to_string())
                            ),
                            start_loc.until(self.location),
                        ));
                    }
                };
                self.next();
                char
            }
            Some(char) if char.is_ascii() => char as u8,
            Some(char) => {
                self.next();
                return Err(Error::new(
                    format!("character `{char}` is not in ASCII range"),
                    start_loc.until(self.location),
                ));
            }
        };
        self.next();
        match self.curr_char {
            Some('\'') => {
                self.next();
                Ok(Token::new(
                    TokenKind::Char(char),
                    start_loc.until(self.location),
                ))
            }
            _ => {
                self.next();
                Err(Error::new(
                    "unterminated char literal".to_string(),
                    start_loc.until(self.location),
                ))
            }
        }
    }

    fn make_number(&mut self) -> Result<Token<'src>> {
        let start_loc = self.location;

        if self.curr_char == Some('0') && self.next_char == Some('x') {
            self.next();
            self.next();
            let start_hex = self.location.byte_idx;

            if !self.curr_char.map_or(false, |c| c.is_ascii_hexdigit()) {
                self.next();
                return Err(Error::new(
                    "expected at least one hexadecimal digit".to_string(),
                    start_loc.until(self.location),
                ));
            }

            while self
                .curr_char
                .map_or(false, |c| c.is_ascii_hexdigit() || c == '_')
            {
                self.next();
            }

            let num = match i64::from_str_radix(
                &self.input[start_hex..self.location.byte_idx].replace('_', ""),
                16,
            ) {
                Ok(num) => num,
                Err(_) => {
                    return Err(Error::new(
                        "integer too large for 64 bits".to_string(),
                        start_loc.until(self.location),
                    ))
                }
            };

            return Ok(TokenKind::Int(num).spanned(start_loc.until(self.location)));
        }

        while self
            .curr_char
            .map_or(false, |c| c.is_ascii_digit() || c == '_')
        {
            self.next();
        }

        match self.curr_char {
            Some('.') => {
                self.next();

                if !self.curr_char.map_or(false, |c| c.is_ascii_digit()) {
                    let err_start = self.location;
                    self.next();
                    return Err(Error::new(
                        format!(
                            "expected digit, found `{}`",
                            self.curr_char.map_or("EOF".to_string(), |c| c.to_string())
                        ),
                        err_start.until(self.location),
                    ));
                }

                while self
                    .curr_char
                    .map_or(false, |c| c.is_ascii_digit() || c == '_')
                {
                    self.next();
                }

                let float = self.input[start_loc.byte_idx..self.location.byte_idx]
                    .replace('_', "")
                    .parse()
                    .expect("The grammar guarantees correctly formed float literals");
                Ok(Token::new(
                    TokenKind::Float(float),
                    start_loc.until(self.location),
                ))
            }
            Some('f') => {
                let float = self.input[start_loc.byte_idx..self.location.byte_idx]
                    .replace('_', "")
                    .parse()
                    .expect("The grammar guarantees correctly formed float literals");
                self.next();
                Ok(Token::new(
                    TokenKind::Float(float),
                    start_loc.until(self.location),
                ))
            }
            _ => {
                let int = match self.input[start_loc.byte_idx..self.location.byte_idx]
                    .replace('_', "")
                    .parse()
                {
                    Ok(value) => value,
                    Err(_) => {
                        return Err(Error::new(
                            "integer too large for 64 bits".to_string(),
                            start_loc.until(self.location),
                        ))
                    }
                };
                Ok(Token::new(
                    TokenKind::Int(int),
                    start_loc.until(self.location),
                ))
            }
        }
    }

    fn make_name(&mut self) -> Token<'src> {
        let start_loc = self.location;
        while self.curr_char.map_or(false, |c| {
            c.is_ascii_alphabetic() || c.is_ascii_digit() || c == '_'
        }) {
            self.next()
        }
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
        kind.spanned(start_loc.until(self.location))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_tokens() {
        let tests = [
            // Chars
            ("'a'", Ok(TokenKind::Char(b'a').spanned(span!(0..3)))),
            ("'*'", Ok(TokenKind::Char(b'*').spanned(span!(0..3)))),
            ("'_'", Ok(TokenKind::Char(b'_').spanned(span!(0..3)))),
            (r"'\'", Err("unterminated char literal")),
            (r"'\\'", Ok(TokenKind::Char(b'\\').spanned(span!(0..4)))),
            (r"'\a'", Err("expected escape character, found a")),
            (r"'\x1b'", Ok(TokenKind::Char(b'\x1b').spanned(span!(0..6)))),
            (r"'\x1b1'", Err("unterminated char literal")),
            // Keywords
            ("true", Ok(TokenKind::True.spanned(span!(0..4)))),
            ("false", Ok(TokenKind::False.spanned(span!(0..5)))),
            ("fn", Ok(TokenKind::Fn.spanned(span!(0..2)))),
            ("let", Ok(TokenKind::Let.spanned(span!(0..3)))),
            ("mut", Ok(TokenKind::Mut.spanned(span!(0..3)))),
            ("return", Ok(TokenKind::Return.spanned(span!(0..6)))),
            ("if", Ok(TokenKind::If.spanned(span!(0..2)))),
            ("else", Ok(TokenKind::Else.spanned(span!(0..4)))),
            ("as", Ok(TokenKind::As.spanned(span!(0..2)))),
            // Identifiers
            ("foo", Ok(TokenKind::Ident("foo").spanned(span!(0..3)))),
            ("_foo", Ok(TokenKind::Ident("_foo").spanned(span!(0..4)))),
            ("f_0o", Ok(TokenKind::Ident("f_0o").spanned(span!(0..4)))),
            // Numbers
            ("1", Ok(TokenKind::Int(1).spanned(span!(0..1)))),
            ("0x1b", Ok(TokenKind::Int(0x1b).spanned(span!(0..4)))),
            ("42", Ok(TokenKind::Int(42).spanned(span!(0..2)))),
            ("42f", Ok(TokenKind::Float(42.0).spanned(span!(0..3)))),
            ("3.1", Ok(TokenKind::Float(3.1).spanned(span!(0..3)))),
            (
                "42.12345678",
                Ok(TokenKind::Float(42.12345678).spanned(span!(0..11))),
            ),
            ("42.69", Ok(TokenKind::Float(42.69).spanned(span!(0..5)))),
            // Parenthesis
            ("(", Ok(TokenKind::LParen.spanned(span!(0..1)))),
            (")", Ok(TokenKind::RParen.spanned(span!(0..1)))),
            ("{", Ok(TokenKind::LBrace.spanned(span!(0..1)))),
            ("}", Ok(TokenKind::RBrace.spanned(span!(0..1)))),
            // Punctuation and delimiters
            ("->", Ok(TokenKind::Arrow.spanned(span!(0..2)))),
            (",", Ok(TokenKind::Comma.spanned(span!(0..1)))),
            (":", Ok(TokenKind::Colon.spanned(span!(0..1)))),
            (";", Ok(TokenKind::Semicolon.spanned(span!(0..1)))),
            // Operators
            ("!", Ok(TokenKind::Not.spanned(span!(0..1)))),
            ("-", Ok(TokenKind::Minus.spanned(span!(0..1)))),
            ("+", Ok(TokenKind::Plus.spanned(span!(0..1)))),
            ("*", Ok(TokenKind::Star.spanned(span!(0..1)))),
            ("/", Ok(TokenKind::Slash.spanned(span!(0..1)))),
            ("%", Ok(TokenKind::Percent.spanned(span!(0..1)))),
            ("**", Ok(TokenKind::Pow.spanned(span!(0..2)))),
            ("==", Ok(TokenKind::Eq.spanned(span!(0..2)))),
            ("!=", Ok(TokenKind::Neq.spanned(span!(0..2)))),
            ("<", Ok(TokenKind::Lt.spanned(span!(0..1)))),
            (">", Ok(TokenKind::Gt.spanned(span!(0..1)))),
            ("<=", Ok(TokenKind::Lte.spanned(span!(0..2)))),
            (">=", Ok(TokenKind::Gte.spanned(span!(0..2)))),
            ("<<", Ok(TokenKind::Shl.spanned(span!(0..2)))),
            (">>", Ok(TokenKind::Shr.spanned(span!(0..2)))),
            ("|", Ok(TokenKind::BitOr.spanned(span!(0..1)))),
            ("&", Ok(TokenKind::BitAnd.spanned(span!(0..1)))),
            ("^", Ok(TokenKind::BitXor.spanned(span!(0..1)))),
            ("&&", Ok(TokenKind::And.spanned(span!(0..2)))),
            ("||", Ok(TokenKind::Or.spanned(span!(0..2)))),
            // Assignments
            ("=", Ok(TokenKind::Assign.spanned(span!(0..1)))),
            ("+=", Ok(TokenKind::PlusAssign.spanned(span!(0..2)))),
            ("-=", Ok(TokenKind::MinusAssign.spanned(span!(0..2)))),
            ("*=", Ok(TokenKind::MulAssign.spanned(span!(0..2)))),
            ("/=", Ok(TokenKind::DivAssign.spanned(span!(0..2)))),
            ("%=", Ok(TokenKind::RemAssign.spanned(span!(0..2)))),
            ("**=", Ok(TokenKind::PowAssign.spanned(span!(0..3)))),
            ("<<=", Ok(TokenKind::ShlAssign.spanned(span!(0..3)))),
            (">>=", Ok(TokenKind::ShrAssign.spanned(span!(0..3)))),
            ("|=", Ok(TokenKind::BitOrAssign.spanned(span!(0..2)))),
            ("&=", Ok(TokenKind::BitAndAssign.spanned(span!(0..2)))),
            ("^=", Ok(TokenKind::BitXorAssign.spanned(span!(0..2)))),
        ];
        println!();
        for (input, expected) in tests {
            let mut lexer = Lexer::new(input);
            let res = lexer.next_token();
            match (res, expected) {
                (Ok(_), Err(expected)) => panic!("Expected error: {:?}, got none", expected),
                (Err(err), Ok(_)) => panic!("Unexpected error: {:?}", err),
                (Err(got), Err(expected)) => assert_eq!(expected, got.message),
                (Ok(got), Ok(expected)) => {
                    match got.kind {
                        TokenKind::Char(ch) => {
                            println!("found char: {} ({ch})", got.kind)
                        }
                        _ => println!("{:?}", got),
                    }
                    assert_eq!(expected, got)
                }
            }
        }
    }

    impl<'src> Iterator for Lexer<'src> {
        type Item = Result<Token<'src>>;

        fn next(&mut self) -> Option<Self::Item> {
            match self.next_token() {
                Ok(Token {
                    kind: TokenKind::Eof,
                    span: _,
                }) => None,
                item => Some(item),
            }
        }
    }

    #[test]
    fn call_expr() {
        let lexer = Lexer::new("exit(1 + 3);");
        assert_eq!(
            lexer.collect::<Result<Vec<_>>>(),
            Ok(vec![
                TokenKind::Ident("exit").spanned(span!(0..4)),
                TokenKind::LParen.spanned(span!(4..5)),
                TokenKind::Int(1).spanned(span!(5..6)),
                TokenKind::Plus.spanned(span!(7..8)),
                TokenKind::Int(3).spanned(span!(9..10)),
                TokenKind::RParen.spanned(span!(10..11)),
                TokenKind::Semicolon.spanned(span!(11..12)),
            ])
        );
    }
}

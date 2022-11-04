use std::mem;

use crate::{ast::*, Error, Lex, Location, Result, Span, Token, TokenKind};

pub struct Parser<'src, Lexer: Lex<'src>> {
    lexer: Lexer,
    prev_tok: Token<'src>,
    curr_tok: Token<'src>,
    errors: Vec<Error>,
}

impl<'src, Lexer: Lex<'src>> Parser<'src, Lexer> {
    /// Creates a new Parser
    pub fn new(lexer: Lexer) -> Self {
        Self {
            lexer,
            // initialize with dummy Eof tokens
            prev_tok: Token::default(),
            curr_tok: Token::default(),
            errors: vec![],
        }
    }

    /// Consumes this parser and tries to parse a [`Program`].
    ///
    /// # Returns
    /// This function returns a tuple of a `Result<Program>`
    /// and a `Vec<Error>`. Parsing can be
    /// - successful: `(Some(Program), [])`
    /// - partially successful: `(Some(Program), [..errors])`
    /// - unsuccessful: `(Err(fatal_error), [..errors])`
    pub fn parse(mut self) -> (Result<ParsedProgram<'src>>, Vec<Error>) {
        if let Err(err) = self.next() {
            return (Err(err), self.errors);
        }

        let program = match self.program() {
            Ok(program) => program,
            Err(err) => return (Err(err), self.errors),
        };

        if self.curr_tok.kind != TokenKind::Eof {
            self.errors.push(Error::new(
                format!("expected EOF, found `{}`", self.curr_tok.kind),
                self.curr_tok.span,
            ))
        }

        (Ok(program), self.errors)
    }

    // moves cursor to next token
    fn next(&mut self) -> Result<()> {
        // swap prev_tok and curr_tok in memory so that what was curr_tok is now prev_tok
        mem::swap(&mut self.prev_tok, &mut self.curr_tok);
        // overwrite curr_tok (which is now what prev_tok was) with the next token from the lexer
        self.curr_tok = self.lexer.next_token()?;

        Ok(())
    }

    // expects the curr_tok to be of the specified kind
    fn expect(&mut self, kind: TokenKind) -> Result<()> {
        if self.curr_tok.kind != kind {
            return Err(Error::new(
                format!("expected `{kind}`, found `{}`", self.curr_tok.kind),
                self.curr_tok.span,
            ));
        }
        self.next()?;
        Ok(())
    }

    // expects the curr_tok to be an identifier and returns its name if this is the case
    fn expect_ident(&mut self) -> Result<ParsedIdent<'src>> {
        match self.curr_tok.kind {
            TokenKind::Ident(value) => {
                let ident = ParsedIdent {
                    span: self.curr_tok.span,
                    annotation: (),
                    value,
                };
                self.next()?;
                Ok(ident)
            }
            _ => Err(Error::new(
                format!("expected identifier, found `{}`", self.curr_tok.kind),
                self.curr_tok.span,
            )),
        }
    }

    // expects curr_tok to be the specified token kind and adds a soft error otherwise
    fn expect_recoverable(&mut self, kind: TokenKind, message: impl Into<String>) -> Result<()> {
        if self.curr_tok.kind != kind {
            self.errors
                .push(Error::new(message.into(), self.curr_tok.span));
        } else {
            self.next()?;
        }
        Ok(())
    }

    //////////////////////////

    fn program(&mut self) -> Result<ParsedProgram<'src>> {
        let start_loc = self.curr_tok.span.start;
        let mut functions = vec![];

        while self.curr_tok.kind != TokenKind::Eof {
            functions.push(self.function_definition()?);
        }

        Ok(Program {
            span: start_loc.until(self.prev_tok.span.end),
            functions,
        })
    }

    fn type_(&mut self) -> Result<ParsedType<'src>> {
        let start_loc = self.curr_tok.span.start;
        let type_ = match self.curr_tok.kind {
            TokenKind::Ident("int") => TypeKind::Int,
            TokenKind::Ident("float") => TypeKind::Float,
            TokenKind::Ident("bool") => TypeKind::Bool,
            TokenKind::Ident("char") => TypeKind::Char,
            TokenKind::Ident(ident) => {
                self.errors.push(Error::new(
                    format!("unknown type `{ident}`"),
                    self.curr_tok.span,
                ));
                TypeKind::Unknown
            }
            TokenKind::LParen => {
                self.next()?;
                self.expect_recoverable(TokenKind::RParen, "missing closing parenthesis")?;
                return Ok(ParsedType {
                    span: start_loc.until(self.curr_tok.span.end),
                    annotation: (),
                    value: TypeKind::Unit,
                });
            }
            invalid => {
                return Err(Error::new(
                    format!("expected a type, found `{invalid}`"),
                    self.curr_tok.span,
                ));
            }
        };
        self.next()?;
        Ok(ParsedType {
            span: start_loc.until(self.prev_tok.span.end),
            annotation: (),
            value: type_,
        })
    }

    fn function_definition(&mut self) -> Result<ParsedFunctionDefinition<'src>> {
        let start_loc = self.curr_tok.span.start;

        self.expect(TokenKind::Fn)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LParen)?;

        let mut params = vec![];
        if !matches!(self.curr_tok.kind, TokenKind::RParen | TokenKind::Eof) {
            params.push(self.parameter()?);
            while self.curr_tok.kind == TokenKind::Comma {
                self.next()?;
                if matches!(self.curr_tok.kind, TokenKind::RParen | TokenKind::Eof) {
                    break;
                }
                params.push(self.parameter()?);
            }
        }

        let rparen_loc = self.curr_tok.span.start;
        self.expect_recoverable(TokenKind::RParen, "missing closing parenthesis")?;

        let return_type = match self.curr_tok.kind {
            TokenKind::Arrow => {
                self.next()?;
                self.type_()?
            }
            _ => ParsedType {
                span: rparen_loc.until(self.curr_tok.span.end),
                annotation: (),
                value: TypeKind::Unit,
            },
        };

        let block = self.block()?;

        Ok(FunctionDefinition {
            span: start_loc.until(self.prev_tok.span.end),
            annotation: (),
            name,
            params,
            return_type,
            block,
        })
    }

    fn parameter(&mut self) -> Result<(ParsedIdent<'src>, ParsedType<'src>)> {
        let name = self.expect_ident()?;
        self.expect(TokenKind::Colon)?;
        let type_ = self.type_()?;
        Ok((name, type_))
    }

    fn block(&mut self) -> Result<ParsedBlock<'src>> {
        let start_loc = self.curr_tok.span.start;

        self.expect(TokenKind::LBrace)?;

        let mut stmts = vec![];
        while !matches!(self.curr_tok.kind, TokenKind::RBrace | TokenKind::Eof) {
            stmts.push(self.statement()?);
        }

        self.expect_recoverable(TokenKind::RBrace, "missing closing brace")?;

        Ok(Block {
            span: start_loc.until(self.prev_tok.span.end),
            annotation: (),
            stmts,
        })
    }

    fn statement(&mut self) -> Result<ParsedStatement<'src>> {
        Ok(match self.curr_tok.kind {
            TokenKind::Let => Statement::Let(self.let_stmt()?),
            TokenKind::Return => Statement::Return(self.return_stmt()?),
            _ => Statement::Expr(self.expr_stmt()?),
        })
    }

    fn let_stmt(&mut self) -> Result<ParsedLetStmt<'src>> {
        let start_loc = self.curr_tok.span.start;

        // skip let token: this function is only called when self.curr_tok.kind == TokenKind::Let
        self.next()?;

        let mutable = self.curr_tok.kind == TokenKind::Mut;
        if mutable {
            self.next()?;
        }

        let name = self.expect_ident()?;

        let type_ = match self.curr_tok.kind {
            TokenKind::Colon => {
                self.next()?;
                Some(self.type_()?)
            }
            _ => None,
        };

        self.expect(TokenKind::Assign)?;
        let expr = self.expression(0)?;
        self.expect_recoverable(TokenKind::Semicolon, "missing semicolon after statement")?;

        Ok(LetStmt {
            span: start_loc.until(self.prev_tok.span.end),
            annotation: (),
            mutable,
            type_,
            name,
            expr,
        })
    }

    fn return_stmt(&mut self) -> Result<ParsedReturnStmt<'src>> {
        let start_loc = self.curr_tok.span.start;

        // skip return token: this function is only called when self.curr_tok.kind == TokenKind::Return
        self.next()?;

        let expr = match self.curr_tok.kind {
            TokenKind::Semicolon => None,
            _ => Some(self.expression(0)?),
        };

        self.expect_recoverable(TokenKind::Semicolon, "missing semicolon after statement")?;

        Ok(ReturnStmt {
            span: start_loc.until(self.prev_tok.span.end),
            annotation: (),
            expr,
        })
    }

    fn expr_stmt(&mut self) -> Result<ParsedExprStmt<'src>> {
        let start_loc = self.curr_tok.span.start;

        let (expr, with_block) = match self.curr_tok.kind {
            TokenKind::If => (Expression::If(self.if_expr()?.into()), true),
            TokenKind::LBrace => (Expression::Block(self.block()?.into()), true),
            _ => (self.expression(0)?, false),
        };

        match (self.curr_tok.kind, with_block) {
            (TokenKind::Semicolon, true) => self.next()?,
            (TokenKind::Semicolon, false) => self.next()?,
            (_, true) => {}
            (_, false) => self.errors.push(Error::new(
                "missing semicolon after statement".to_string(),
                self.curr_tok.span,
            )),
        }

        Ok(ExprStmt {
            span: start_loc.until(self.prev_tok.span.end),
            annotation: (),
            expr,
        })
    }

    fn expression(&mut self, prec: u8) -> Result<ParsedExpression<'src>> {
        let start_loc = self.curr_tok.span.start;

        let mut lhs = match self.curr_tok.kind {
            TokenKind::LBrace => Expression::Block(self.block()?.into()),
            TokenKind::If => Expression::If(self.if_expr()?.into()),
            TokenKind::Int(num) => Expression::Int(self.atom(num)?),
            TokenKind::Float(num) => Expression::Float(self.atom(num)?),
            TokenKind::True => Expression::Bool(self.atom(true)?),
            TokenKind::False => Expression::Bool(self.atom(false)?),
            TokenKind::Ident(ident) => Expression::Ident(self.atom(ident)?),
            TokenKind::Not => Expression::Prefix(self.prefix_expr(PrefixOp::Not)?.into()),
            TokenKind::Minus => Expression::Prefix(self.prefix_expr(PrefixOp::Neg)?.into()),
            TokenKind::LParen => Expression::Grouped(self.grouped_expr()?),
            invalid => {
                return Err(Error::new(
                    format!("expected an expression, found `{invalid}`"),
                    self.curr_tok.span,
                ));
            }
        };

        while self.curr_tok.kind.prec().0 > prec {
            lhs = match self.curr_tok.kind {
                TokenKind::Plus => self.infix_expr(start_loc, lhs, InfixOp::Plus)?,
                TokenKind::Minus => self.infix_expr(start_loc, lhs, InfixOp::Minus)?,
                TokenKind::Star => self.infix_expr(start_loc, lhs, InfixOp::Mul)?,
                TokenKind::Slash => self.infix_expr(start_loc, lhs, InfixOp::Div)?,
                TokenKind::Percent => self.infix_expr(start_loc, lhs, InfixOp::Rem)?,
                TokenKind::Pow => self.infix_expr(start_loc, lhs, InfixOp::Pow)?,
                TokenKind::Eq => self.infix_expr(start_loc, lhs, InfixOp::Eq)?,
                TokenKind::Neq => self.infix_expr(start_loc, lhs, InfixOp::Neq)?,
                TokenKind::Lt => self.infix_expr(start_loc, lhs, InfixOp::Lt)?,
                TokenKind::Gt => self.infix_expr(start_loc, lhs, InfixOp::Gt)?,
                TokenKind::Lte => self.infix_expr(start_loc, lhs, InfixOp::Lte)?,
                TokenKind::Gte => self.infix_expr(start_loc, lhs, InfixOp::Gte)?,
                TokenKind::Shl => self.infix_expr(start_loc, lhs, InfixOp::Shl)?,
                TokenKind::Shr => self.infix_expr(start_loc, lhs, InfixOp::Shr)?,
                TokenKind::BitOr => self.infix_expr(start_loc, lhs, InfixOp::BitOr)?,
                TokenKind::BitAnd => self.infix_expr(start_loc, lhs, InfixOp::BitAnd)?,
                TokenKind::BitXor => self.infix_expr(start_loc, lhs, InfixOp::BitXor)?,
                TokenKind::And => self.infix_expr(start_loc, lhs, InfixOp::And)?,
                TokenKind::Or => self.infix_expr(start_loc, lhs, InfixOp::Or)?,
                TokenKind::Assign => self.assign_expr(start_loc, lhs, AssignOp::Basic)?,
                TokenKind::PlusAssign => self.assign_expr(start_loc, lhs, AssignOp::Plus)?,
                TokenKind::MinusAssign => self.assign_expr(start_loc, lhs, AssignOp::Minus)?,
                TokenKind::MulAssign => self.assign_expr(start_loc, lhs, AssignOp::Mul)?,
                TokenKind::DivAssign => self.assign_expr(start_loc, lhs, AssignOp::Div)?,
                TokenKind::RemAssign => self.assign_expr(start_loc, lhs, AssignOp::Rem)?,
                TokenKind::PowAssign => self.assign_expr(start_loc, lhs, AssignOp::Pow)?,
                TokenKind::ShlAssign => self.assign_expr(start_loc, lhs, AssignOp::Shl)?,
                TokenKind::ShrAssign => self.assign_expr(start_loc, lhs, AssignOp::Shr)?,
                TokenKind::BitOrAssign => self.assign_expr(start_loc, lhs, AssignOp::BitOr)?,
                TokenKind::BitAndAssign => self.assign_expr(start_loc, lhs, AssignOp::BitAnd)?,
                TokenKind::BitXorAssign => self.assign_expr(start_loc, lhs, AssignOp::BitXor)?,
                TokenKind::LParen => self.call_expr(start_loc, lhs)?,
                TokenKind::As => self.cast_expr(start_loc, lhs)?,
                _ => return Ok(lhs),
            };
        }

        Ok(lhs)
    }

    fn if_expr(&mut self) -> Result<ParsedIfExpr<'src>> {
        let start_loc = self.curr_tok.span.start;

        // skip if token: this function is only called when self.curr_tok.kind == TokenKind::If
        self.next()?;

        let cond = self.expression(0)?;
        let then_block = self.block()?;
        let else_block = match self.curr_tok.kind {
            TokenKind::Else => {
                self.next()?;
                Some(match self.curr_tok.kind {
                    TokenKind::If => {
                        let if_expr = self.if_expr()?;
                        Block {
                            span: if_expr.span,
                            annotation: (),
                            stmts: vec![Statement::Expr(ExprStmt {
                                span: if_expr.span,
                                annotation: (),
                                expr: Expression::If(if_expr.into()),
                            })],
                        }
                    }
                    TokenKind::LBrace => self.block()?,
                    invalid => {
                        return Err(Error::new(
                            format!(
                                "expected either `if` or block after `else`, found `{invalid}`"
                            ),
                            self.curr_tok.span,
                        ));
                    }
                })
            }
            _ => None,
        };

        Ok(IfExpr {
            span: start_loc.until(self.prev_tok.span.end),
            annotation: (),
            cond,
            then_block,
            else_block,
        })
    }

    fn atom<T>(&mut self, value: T) -> Result<ParsedAtom<T>> {
        let start_loc = self.curr_tok.span.start;
        self.next()?;
        Ok(Atom {
            span: start_loc.until(self.prev_tok.span.end),
            annotation: (),
            value,
        })
    }

    fn prefix_expr(&mut self, op: PrefixOp) -> Result<ParsedPrefixExpr<'src>> {
        let start_loc = self.curr_tok.span.start;

        // skip the operator token
        self.next()?;

        // PrefixExpr precedence is 29, higher than all InfixExpr precedences
        let expr = self.expression(29)?;
        Ok(PrefixExpr {
            span: start_loc.until(self.prev_tok.span.end),
            annotation: (),
            op,
            expr,
        })
    }

    fn grouped_expr(&mut self) -> Result<ParsedAtom<Box<ParsedExpression<'src>>>> {
        let start_loc = self.curr_tok.span.start;
        // skip the opening parenthesis
        self.next()?;

        let expr = self.expression(0)?;
        self.expect_recoverable(TokenKind::RParen, "missing closing parenthesis")?;

        Ok(Atom {
            span: start_loc.until(self.prev_tok.span.end),
            annotation: (),
            value: expr.into(),
        })
    }

    fn infix_expr(
        &mut self,
        start_loc: Location,
        lhs: ParsedExpression<'src>,
        op: InfixOp,
    ) -> Result<ParsedExpression<'src>> {
        let right_prec = self.curr_tok.kind.prec().1;
        self.next()?;
        let rhs = self.expression(right_prec)?;

        Ok(Expression::Infix(
            InfixExpr {
                span: start_loc.until(self.prev_tok.span.end),
                annotation: (),
                lhs,
                op,
                rhs,
            }
            .into(),
        ))
    }

    fn assign_expr(
        &mut self,
        start_loc: Location,
        lhs: ParsedExpression<'src>,
        op: AssignOp,
    ) -> Result<ParsedExpression<'src>> {
        let assignee = match lhs {
            Expression::Ident(item) => item,
            _ => {
                self.errors.push(Error::new(
                    "left hand side of assignment must be an identifier".to_string(),
                    self.curr_tok.span,
                ));
                Atom {
                    span: Span::default(),
                    annotation: (),
                    value: "",
                }
            }
        };

        let right_prec = self.curr_tok.kind.prec().1;
        self.next()?;
        let expr = self.expression(right_prec)?;

        Ok(Expression::Assign(
            AssignExpr {
                span: start_loc.until(self.prev_tok.span.end),
                annotation: (),
                assignee,
                op,
                expr,
            }
            .into(),
        ))
    }

    fn call_expr(
        &mut self,
        start_loc: Location,
        expr: ParsedExpression<'src>,
    ) -> Result<ParsedExpression<'src>> {
        // skip opening parenthesis
        self.next()?;

        let mut args = vec![];
        if !matches!(self.curr_tok.kind, TokenKind::RParen | TokenKind::Eof) {
            args.push(self.expression(0)?);

            while self.curr_tok.kind == TokenKind::Comma {
                self.next()?;
                if let TokenKind::RParen | TokenKind::Eof = self.curr_tok.kind {
                    break;
                }
                args.push(self.expression(0)?);
            }
        }

        self.expect_recoverable(TokenKind::RParen, "missing closing parenthesis")?;

        Ok(Expression::Call(
            CallExpr {
                span: start_loc.until(self.prev_tok.span.end),
                annotation: (),
                expr,
                args,
            }
            .into(),
        ))
    }

    fn cast_expr(
        &mut self,
        start_loc: Location,
        expr: ParsedExpression<'src>,
    ) -> Result<ParsedExpression<'src>> {
        // skip `as` token
        self.next()?;

        let type_ = self.type_()?;

        Ok(Expression::Cast(
            CastExpr {
                span: start_loc.until(self.prev_tok.span.end),
                annotation: (),
                expr,
                type_,
            }
            .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Impl Lex trait for any iterator over tokens
    impl<'src, T> Lex<'src> for T
    where
        T: Iterator<Item = Token<'src>>,
    {
        fn next_token(&mut self) -> Result<Token<'src>> {
            Ok(self.next().unwrap_or_default())
        }
    }

    /// Parses the `tokens` into an [`Expression`] and asserts equality with `tree`
    /// without any errors
    fn expr_test(
        tokens: impl IntoIterator<Item = Token<'static>>,
        tree: ParsedExpression<'static>,
    ) -> Result<()> {
        let mut parser = Parser::new(tokens.into_iter());
        parser.next()?;
        assert_eq!(dbg!(parser.expression(0)?), dbg!(tree));
        assert!(dbg!(parser.errors).is_empty());
        Ok(())
    }

    /// Parses the `tokens` into a [`Statement`] and asserts equality with `tree`
    /// without any errors
    fn stmt_test(
        tokens: impl IntoIterator<Item = Token<'static>>,
        tree: ParsedStatement<'static>,
    ) -> Result<()> {
        let mut parser = Parser::new(tokens.into_iter());
        parser.next()?;
        assert_eq!(dbg!(parser.statement()?), dbg!(tree));
        assert!(dbg!(parser.errors).is_empty());
        Ok(())
    }

    /// Parses the `tokens` into a [`Program`] and asserts equality with `tree`
    /// without any errors
    fn program_test(
        tokens: impl IntoIterator<Item = Token<'static>>,
        tree: ParsedProgram<'static>,
    ) -> Result<()> {
        let parser = Parser::new(tokens.into_iter());
        let (res, errors) = parser.parse();
        assert_eq!(dbg!(res?), dbg!(tree));
        assert!(dbg!(errors).is_empty());
        Ok(())
    }

    #[test]
    fn arithmetic_expressions() -> Result<()> {
        // 3-2-1
        expr_test(
            tokens![
                Int(3) @ 0..1,
                Minus @ 1..2,
                Int(2) @ 2..3,
                Minus @ 3..4,
                Int(1) @ 4..5,
            ],
            tree! {
                (InfixExpr @ 0..5,
                    lhs: (InfixExpr @ 0..3,
                        lhs: (Int @ 0..1, 3),
                        op: Minus,
                        rhs: (Int @ 2..3, 2)),
                    op: Minus,
                    rhs: (Int @ 4..5, 1))
            },
        )?;

        // 1+2*3
        expr_test(
            tokens![
                Int(1) @ 0..1,
                Plus @ 1..2,
                Int(2) @ 2..3,
                Star @ 3..4,
                Int(3) @ 4..5,
            ],
            tree! {
                (InfixExpr @ 0..5,
                    lhs: (Int @ 0..1, 1),
                    op: Plus,
                    rhs: (InfixExpr @ 2..5,
                        lhs: (Int @ 2..3, 2),
                        op: Mul,
                        rhs: (Int @ 4..5, 3)))
            },
        )?;

        // 2**3**4
        expr_test(
            tokens![
                Int(2) @ 0..1,
                Pow @ 1..3,
                Int(3) @ 3..4,
                Pow @ 4..6,
                Int(4) @ 6..7,
            ],
            tree! {
                (InfixExpr @ 0..7,
                    lhs: (Int @ 0..1, 2),
                    op: Pow,
                    rhs: (InfixExpr @ 3..7,
                        lhs: (Int @ 3..4, 3),
                        op: Pow,
                        rhs: (Int @ 6..7, 4)))
            },
        )?;

        Ok(())
    }

    #[test]
    fn assignment_expressions() -> Result<()> {
        // a=1
        expr_test(
            tokens![
                Ident("a") @ 0..1,
                Assign @ 1..2,
                Int(1) @ 2..3,
            ],
            tree! {
                (AssignExpr @ 0..3,
                    assignee: (Atom @ 0..1, "a"),
                    op: Basic,
                    expr: (Int @ 2..3, 1))
            },
        )?;

        // answer += 42.0 - 0f
        expr_test(
            tokens![
                Ident("answer") @ 0..6,
                PlusAssign @ 7..9,
                Float(42.0) @ 10..14,
                Minus @ 15..16,
                Float(0.0) @ 17..19,
            ],
            tree! {
                (AssignExpr @ 0..19,
                    assignee: (Atom @ 0..6, "answer"),
                    op: Plus,
                    expr: (InfixExpr @ 10..19,
                        lhs: (Float @ 10..14, 42.0),
                        op: Minus,
                        rhs: (Float @ 17..19, 0.0)))
            },
        )?;

        Ok(())
    }

    #[test]
    fn let_statement() -> Result<()> {
        // let a=1;
        stmt_test(
            tokens![
                Let @ 0..3,
                Ident("a") @ 4..5,
                Assign @ 5..6,
                Int(1) @ 6..7,
                Semicolon @ 7..8,
            ],
            tree! {
                (LetStmt @ 0..8,
                    mutable: false,
                    name: (Atom @ 4..5, "a"),
                    type: (None),
                    expr: (Int @ 6..7, 1))
            },
        )?;

        // let mut b = 2;
        stmt_test(
            tokens![
                Let @ 0..3,
                Mut @ 4..7,
                Ident("b") @ 8..9,
                Assign @ 10..11,
                Int(2) @ 12..13,
                Semicolon @ 13..14,
            ],
            tree! {
                (LetStmt @ 0..14,
                    mutable: true,
                    name: (Atom @ 8..9, "b"),
                    type: (None),
                    expr: (Int @ 12..13, 2))
            },
        )?;

        Ok(())
    }

    #[test]
    fn programs() -> Result<()> {
        // fn main() { let a = true; a; }
        program_test(
            tokens![
                Fn @ 0..2,
                Ident("main") @ 3..7,
                LParen @ 7..8,
                RParen @ 8..9,
                LBrace @ 10..11,
                Let @ 12..15,
                Ident("a") @ 16..17,
                Assign @ 18..19,
                True @ 20..24,
                Semicolon @ 24..25,
                Ident("a") @ 26..27,
                Semicolon @ 27..28,
                RBrace @ 29..30,
            ],
            tree! {
                (Program @ 0..30, [
                    (FunctionDefinition @ 0..30,
                        name: (Atom @ 3..7, "main"),
                        params: [],
                        return_type: (Atom @ 8..11, TypeKind::Unit),
                        block: (Block @ 10..30, [
                            (LetStmt @ 12..25,
                                mutable: false,
                                name: (Atom @ 16..17, "a"),
                                type: (None),
                                expr: (Bool @ 20..24, true)),
                            (ExprStmt @ 26..28, (Ident @ 26..27, "a"))]))])
            },
        )?;

        // fn add(left: int, right: int) -> int { return left + right; }
        program_test(
            tokens![
                Fn @ 0..2,
                Ident("add") @ 3..6,
                LParen @ 6..7,
                Ident("left") @ 7..11,
                Colon @ 11..12,
                Ident("int") @ 13..16,
                Comma @ 16..17,
                Ident("right") @ 18..23,
                Colon @ 23..24,
                Ident("int") @ 25..28,
                RParen @ 28..29,
                Arrow @ 30..32,
                Ident("int") @ 33..36,
                LBrace @ 37..38,
                Return @ 39..45,
                Ident("left") @ 46..50,
                Plus @ 51..52,
                Ident("right") @ 53..58,
                Semicolon @ 58..59,
                RBrace @ 60..61,
            ],
            tree! {
                (Program @ 0..61, [
                    (FunctionDefinition @ 0..61,
                        name: (Atom @ 3..6, "add"),
                        params: [
                            ((Atom @ 7..11, "left"), (Atom @ 13..16, TypeKind::Int)),
                            ((Atom @ 18..23, "right"), (Atom @ 25..28, TypeKind::Int))],
                        return_type: (Atom @ 33..36, TypeKind::Int),
                        block: (Block @ 37..61, [
                            (ReturnStmt @ 39..59, (Some(InfixExpr @ 46..58,
                                lhs: (Ident @ 46..50, "left"),
                                op: Plus,
                                rhs: (Ident @ 53..58, "right"))))]))])
            },
        )?;

        Ok(())
    }
}

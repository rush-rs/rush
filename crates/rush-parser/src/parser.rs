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
                self.next()?;
                Ok(ParsedIdent {
                    span: self.curr_tok.span,
                    annotation: (),
                    value,
                })
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

    fn type_(&mut self) -> Result<Type> {
        let type_ = match self.curr_tok.kind {
            TokenKind::Ident("int") => Type::Int,
            TokenKind::Ident("float") => Type::Float,
            TokenKind::Ident("bool") => Type::Bool,
            TokenKind::Ident("char") => Type::Char,
            TokenKind::Ident(ident) => {
                self.errors.push(Error::new(
                    format!("unknown type `{ident}`"),
                    self.curr_tok.span,
                ));
                Type::Unknown
            }
            TokenKind::LParen => {
                self.next()?;
                self.expect_recoverable(TokenKind::RParen, "missing closing parenthesis")?;
                return Ok(Type::Unit);
            }
            invalid => {
                return Err(Error::new(
                    format!("expected a type, found `{invalid}`"),
                    self.curr_tok.span,
                ));
            }
        };
        self.next()?;
        Ok(type_)
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

        self.expect_recoverable(TokenKind::RParen, "missing closing parenthesis")?;

        let return_type = match self.curr_tok.kind {
            TokenKind::Arrow => {
                self.next()?;
                self.type_()?
            }
            _ => Type::Unit,
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

    fn parameter(&mut self) -> Result<(ParsedIdent<'src>, Type)> {
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

    impl<'src, T> Lex<'src> for T
    where
        T: Iterator<Item = Token<'src>>,
    {
        fn next_token(&mut self) -> Result<Token<'src>> {
            Ok(self.next().unwrap_or_default())
        }
    }

    fn expr_test(
        tokens: impl IntoIterator<Item = Token<'static>>,
        tree: ParsedExpression<'static>,
    ) -> Result<()> {
        let mut parser = Parser::new(tokens.into_iter());
        parser.next()?;
        assert_eq!(parser.expression(0)?, tree);
        Ok(())
    }

    #[test]
    fn arithmetic_expressions() -> Result<()> {
        // 3-2-1
        expr_test(
            [
                TokenKind::Int(3).spanned(span!(0..1)),
                TokenKind::Minus.spanned(span!(1..2)),
                TokenKind::Int(2).spanned(span!(2..3)),
                TokenKind::Minus.spanned(span!(3..4)),
                TokenKind::Int(1).spanned(span!(4..5)),
            ],
            Expression::Infix(
                InfixExpr {
                    span: span!(0..5),
                    annotation: (),
                    lhs: Expression::Infix(
                        InfixExpr {
                            span: span!(0..3),
                            annotation: (),
                            lhs: Expression::Int(Atom {
                                span: span!(0..1),
                                annotation: (),
                                value: 3,
                            }),
                            op: InfixOp::Minus,
                            rhs: Expression::Int(Atom {
                                span: span!(2..3),
                                annotation: (),
                                value: 2,
                            }),
                        }
                        .into(),
                    ),
                    op: InfixOp::Minus,
                    rhs: Expression::Int(Atom {
                        span: span!(4..5),
                        annotation: (),
                        value: 1,
                    }),
                }
                .into(),
            ),
        )?;

        // 1+2*3
        expr_test(
            [
                TokenKind::Int(1).spanned(span!(0..1)),
                TokenKind::Plus.spanned(span!(1..2)),
                TokenKind::Int(2).spanned(span!(2..3)),
                TokenKind::Star.spanned(span!(3..4)),
                TokenKind::Int(3).spanned(span!(4..5)),
            ],
            Expression::Infix(
                InfixExpr {
                    span: span!(0..5),
                    annotation: (),
                    lhs: Expression::Int(Atom {
                        span: span!(0..1),
                        annotation: (),
                        value: 1,
                    }),
                    op: InfixOp::Plus,
                    rhs: Expression::Infix(
                        InfixExpr {
                            span: span!(2..5),
                            annotation: (),
                            lhs: Expression::Int(Atom {
                                span: span!(2..3),
                                annotation: (),
                                value: 2,
                            }),
                            op: InfixOp::Mul,
                            rhs: Expression::Int(Atom {
                                span: span!(4..5),
                                annotation: (),
                                value: 3,
                            }),
                        }
                        .into(),
                    ),
                }
                .into(),
            ),
        )?;

        // 2**3**4
        expr_test(
            [
                TokenKind::Int(2).spanned(span!(0..1)),
                TokenKind::Pow.spanned(span!(1..3)),
                TokenKind::Int(3).spanned(span!(3..4)),
                TokenKind::Pow.spanned(span!(4..6)),
                TokenKind::Int(4).spanned(span!(6..7)),
            ],
            Expression::Infix(
                InfixExpr {
                    span: span!(0..7),
                    annotation: (),
                    lhs: Expression::Int(Atom {
                        span: span!(0..1),
                        annotation: (),
                        value: 2,
                    }),
                    op: InfixOp::Pow,
                    rhs: Expression::Infix(
                        InfixExpr {
                            span: span!(3..7),
                            annotation: (),
                            lhs: Expression::Int(Atom {
                                span: span!(3..4),
                                annotation: (),
                                value: 3,
                            }),
                            op: InfixOp::Pow,
                            rhs: Expression::Int(Atom {
                                span: span!(6..7),
                                annotation: (),
                                value: 4,
                            }),
                        }
                        .into(),
                    ),
                }
                .into(),
            ),
        )?;

        Ok(())
    }

    #[test]
    fn assignment_expressions() -> Result<()> {
        // a=1
        expr_test(
            [
                TokenKind::Ident("a").spanned(span!(0..1)),
                TokenKind::Assign.spanned(span!(1..2)),
                TokenKind::Int(1).spanned(span!(2..3)),
            ],
            Expression::Assign(
                AssignExpr {
                    span: span!(0..3),
                    annotation: (),
                    assignee: Atom {
                        span: span!(0..1),
                        annotation: (),
                        value: "a",
                    },
                    op: AssignOp::Basic,
                    expr: Expression::Int(Atom {
                        span: span!(2..3),
                        annotation: (),
                        value: 1,
                    }),
                }
                .into(),
            ),
        )?;

        // answer += 42.0
        expr_test(
            [
                TokenKind::Ident("answer").spanned(span!(0..6)),
                TokenKind::PlusAssign.spanned(span!(7..9)),
                TokenKind::Float(42.0).spanned(span!(10..14)),
            ],
            Expression::Assign(
                AssignExpr {
                    span: span!(0..14),
                    annotation: (),
                    assignee: Atom {
                        span: span!(0..6),
                        annotation: (),
                        value: "answer",
                    },
                    op: AssignOp::Plus,
                    expr: Expression::Float(Atom {
                        span: span!(10..14),
                        annotation: (),
                        value: 42.0,
                    }),
                }
                .into(),
            ),
        )?;

        Ok(())
    }

    // TODO: more tests
}

use std::{fmt::Debug, mem};

use crate::{ast::*, Error, ErrorKind, Lex, Location, Result, Span, Token, TokenKind};

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
            // Initialize with dummy Eof tokens
            prev_tok: TokenKind::Eof.spanned(Span::default()),
            curr_tok: TokenKind::Eof.spanned(Span::default()),
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
    pub fn parse(mut self) -> (Result<Program<'src>>, Vec<Error>) {
        if let Err(err) = self.next() {
            return (Err(err), self.errors);
        }
        let program = match self.program() {
            Ok(program) => program,
            Err(err) => return (Err(err), self.errors),
        };

        if self.curr_tok.kind != TokenKind::Eof {
            self.errors.push(Error::new(
                ErrorKind::Syntax,
                format!("expected EOF, found {}", self.curr_tok.kind),
                self.curr_tok.span,
            ))
        }

        (Ok(program), self.errors)
    }

    // Move cursor to next token
    fn next(&mut self) -> Result<()> {
        // Swap prev_tok and curr_tok in memory so that what was curr_tok is now prev_tok
        mem::swap(&mut self.prev_tok, &mut self.curr_tok);
        // Overwrite curr_tok (which is now what prev_tok was) with the next token from the lexer
        self.curr_tok = self.lexer.next_token()?;

        Ok(())
    }

    fn expect(&mut self, kind: TokenKind) -> Result<()> {
        if self.curr_tok.kind != kind {
            return Err(Error::new(
                ErrorKind::Syntax,
                format!("expected {kind}, found {}", self.curr_tok.kind),
                self.curr_tok.span,
            ));
        }
        self.next()?;
        Ok(())
    }

    fn expect_ident(&mut self) -> Result<&'src str> {
        match self.curr_tok.kind {
            TokenKind::Ident(ident) => {
                self.next()?;
                Ok(ident)
            }
            _ => Err(Error::new(
                ErrorKind::Syntax,
                format!("expected identifier, found {}", self.curr_tok.kind),
                self.curr_tok.span,
            )),
        }
    }

    //////////////////////////

    fn program(&mut self) -> Result<Program<'src>> {
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
                    ErrorKind::Syntax,
                    format!("unknown type `{ident}`"),
                    self.curr_tok.span,
                ));
                Type::Unit
            }
            TokenKind::LParen => {
                self.next()?;
                if self.curr_tok.kind != TokenKind::RParen {
                    self.errors.push(Error::new(
                        ErrorKind::Syntax,
                        "missing closing parenthesis".to_string(),
                        self.curr_tok.span,
                    ));
                } else {
                    self.next()?;
                }
                return Ok(Type::Unit);
            }
            kind => {
                return Err(Error::new(
                    ErrorKind::Syntax,
                    format!("expected a type, found {kind}"),
                    self.curr_tok.span,
                ));
            }
        };
        self.next()?;
        Ok(type_)
    }

    fn function_definition(&mut self) -> Result<FunctionDefinition<'src>> {
        let start_loc = self.curr_tok.span.start;

        self.expect(TokenKind::Fn)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LParen)?;

        let mut params = vec![];
        if !matches!(self.curr_tok.kind, TokenKind::RParen | TokenKind::Eof) {
            params.push(self.parameter()?);
            while self.curr_tok.kind == TokenKind::Comma {
                self.next()?;
                if let TokenKind::RParen | TokenKind::Eof = self.curr_tok.kind {
                    break;
                }
                params.push(self.parameter()?);
            }
        }

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
            name,
            params,
            return_type,
            block,
        })
    }

    fn parameter(&mut self) -> Result<(&'src str, Type)> {
        let name = self.expect_ident()?;
        self.expect(TokenKind::Colon)?;
        let type_ = self.type_()?;
        Ok((name, type_))
    }

    fn block(&mut self) -> Result<Block<'src>> {
        let start_loc = self.curr_tok.span.start;

        self.expect(TokenKind::LBrace)?;

        let mut stmts = vec![];
        while !matches!(self.curr_tok.kind, TokenKind::RBrace | TokenKind::Eof) {
            stmts.push(self.statement()?);
        }

        if self.curr_tok.kind != TokenKind::RBrace {
            self.errors.push(Error::new(
                ErrorKind::Syntax,
                "missing closing brace".to_string(),
                self.curr_tok.span,
            ));
        } else {
            self.next()?;
        }

        Ok(Block {
            span: start_loc.until(self.prev_tok.span.end),
            stmts,
        })
    }

    fn statement(&mut self) -> Result<Statement<'src>> {
        Ok(match self.curr_tok.kind {
            TokenKind::Let => Statement::Let(self.let_stmt()?),
            TokenKind::Return => Statement::Return(self.return_stmt()?),
            _ => Statement::Expr(self.expr_stmt()?),
        })
    }

    fn let_stmt(&mut self) -> Result<LetStmt<'src>> {
        let start_loc = self.curr_tok.span.start;

        // this function is only called when self.curr_tok.kind == TokenKind::Let
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

        if self.curr_tok.kind != TokenKind::Semicolon {
            self.errors.push(Error::new(
                ErrorKind::Syntax,
                "missing semicolon after statement".to_string(),
                self.curr_tok.span,
            ))
        } else {
            self.next()?;
        }

        Ok(LetStmt {
            span: start_loc.until(self.prev_tok.span.end),
            mutable,
            type_,
            name,
            expr,
        })
    }

    fn return_stmt(&mut self) -> Result<ReturnStmt<'src>> {
        let start_loc = self.curr_tok.span.start;

        // this function is only called when self.curr_tok.kind == TokenKind::Return
        self.next()?;

        let expr = match self.curr_tok.kind {
            TokenKind::Semicolon => None,
            _ => Some(self.expression(0)?),
        };

        if self.curr_tok.kind != TokenKind::Semicolon {
            self.errors.push(Error::new(
                ErrorKind::Syntax,
                "missing semicolon after statement".to_string(),
                self.curr_tok.span,
            ))
        } else {
            self.next()?;
        }

        Ok(ReturnStmt {
            span: start_loc.until(self.prev_tok.span.end),
            expr,
        })
    }

    fn expr_stmt(&mut self) -> Result<ExprStmt<'src>> {
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
                ErrorKind::Syntax,
                "missing semicolon after statement".to_string(),
                self.curr_tok.span,
            )),
        }

        Ok(ExprStmt {
            span: start_loc.until(self.prev_tok.span.end),
            expr,
        })
    }

    fn expression(&mut self, prec: u8) -> Result<Expression<'src>> {
        let start_loc = self.curr_tok.span.start;

        let mut lhs = match self.curr_tok.kind {
            TokenKind::LBrace => Expression::Block(self.block()?.into()),
            TokenKind::If => Expression::If(self.if_expr()?.into()),
            TokenKind::Int(num) => Expression::Int(self.atom_expr(num)?),
            TokenKind::Float(num) => Expression::Float(self.atom_expr(num)?),
            TokenKind::True => Expression::Bool(self.atom_expr(true)?),
            TokenKind::False => Expression::Bool(self.atom_expr(false)?),
            TokenKind::Ident(ident) => Expression::Ident(self.atom_expr(ident)?),
            TokenKind::Not => Expression::Prefix(self.prefix_expr(PrefixOp::Not)?.into()),
            TokenKind::Minus => Expression::Prefix(self.prefix_expr(PrefixOp::Neg)?.into()),
            TokenKind::LParen => Expression::Grouped(self.expression(0)?.into()),
            kind => {
                return Err(Error::new(
                    ErrorKind::Syntax,
                    format!("expected an expression, found {kind}"),
                    self.curr_tok.span,
                ));
            }
        };

        while self.curr_tok.kind.prec().0 > prec {
            lhs = match self.curr_tok.kind {
                TokenKind::Plus => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Plus)?.into())
                }
                TokenKind::Minus => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Minus)?.into())
                }
                TokenKind::Star => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Mul)?.into())
                }
                TokenKind::Slash => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Div)?.into())
                }
                TokenKind::Percent => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Rem)?.into())
                }
                TokenKind::Pow => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Pow)?.into())
                }
                TokenKind::Eq => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Eq)?.into())
                }
                TokenKind::Neq => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Neq)?.into())
                }
                TokenKind::Lt => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Lt)?.into())
                }
                TokenKind::Gt => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Gt)?.into())
                }
                TokenKind::Lte => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Lte)?.into())
                }
                TokenKind::Gte => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Gte)?.into())
                }
                TokenKind::Shl => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Shl)?.into())
                }
                TokenKind::Shr => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Shr)?.into())
                }
                TokenKind::BitOr => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::BitOr)?.into())
                }
                TokenKind::BitAnd => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::BitAnd)?.into())
                }
                TokenKind::BitXor => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::BitXor)?.into())
                }
                TokenKind::And => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::And)?.into())
                }
                TokenKind::Or => {
                    Expression::Infix(self.infix_expr(start_loc, lhs, InfixOp::Or)?.into())
                }
                TokenKind::Assign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::Basic)?.into())
                }
                TokenKind::PlusAssign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::Plus)?.into())
                }
                TokenKind::MinusAssign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::Minus)?.into())
                }
                TokenKind::MulAssign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::Mul)?.into())
                }
                TokenKind::DivAssign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::Div)?.into())
                }
                TokenKind::RemAssign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::Rem)?.into())
                }
                TokenKind::PowAssign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::Pow)?.into())
                }
                TokenKind::ShlAssign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::Shl)?.into())
                }
                TokenKind::ShrAssign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::Shr)?.into())
                }
                TokenKind::BitOrAssign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::BitOr)?.into())
                }
                TokenKind::BitAndAssign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::BitAnd)?.into())
                }
                TokenKind::BitXorAssign => {
                    Expression::Assign(self.assign_expr(start_loc, lhs, AssignOp::BitXor)?.into())
                }
                TokenKind::LParen => Expression::Call(self.call_expr(start_loc, lhs)?.into()),
                TokenKind::As => Expression::Cast(self.cast_expr(start_loc, lhs)?.into()),
                _ => return Ok(lhs),
            };
        }

        Ok(lhs)
    }

    fn if_expr(&mut self) -> Result<IfExpr<'src>> {
        let start_loc = self.curr_tok.span.start;

        // this function is only called when self.curr_tok.kind == TokenKind::If
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
                            stmts: vec![Statement::Expr(ExprStmt {
                                span: if_expr.span,
                                expr: Expression::If(Box::new(if_expr)),
                            })],
                        }
                    }
                    TokenKind::LBrace => self.block()?,
                    kind => {
                        return Err(Error::new(
                            ErrorKind::Syntax,
                            format!("expected either `if` or block after `else`, found {kind}"),
                            self.curr_tok.span,
                        ));
                    }
                })
            }
            _ => None,
        };

        Ok(IfExpr {
            span: start_loc.until(self.prev_tok.span.end),
            cond,
            then_block,
            else_block,
        })
    }

    fn atom_expr<T: Debug + Clone + PartialEq>(&mut self, value: T) -> Result<AtomExpr<T>> {
        let start_loc = self.curr_tok.span.start;
        self.next()?;
        Ok(AtomExpr {
            span: start_loc.until(self.prev_tok.span.end),
            value,
        })
    }

    fn prefix_expr(&mut self, op: PrefixOp) -> Result<PrefixExpr<'src>> {
        let start_loc = self.curr_tok.span.start;

        // Skip the operator token
        self.next()?;

        // PrefixExpr precedence is 29, higher than all InfixExpr precedences
        let expr = self.expression(29)?;
        Ok(PrefixExpr {
            span: start_loc.until(self.prev_tok.span.end),
            op,
            expr,
        })
    }

    fn infix_expr(
        &mut self,
        start_loc: Location,
        lhs: Expression<'src>,
        op: InfixOp,
    ) -> Result<InfixExpr<'src>> {
        let right_prec = self.curr_tok.kind.prec().1;
        self.next()?;
        let rhs = self.expression(right_prec)?;

        Ok(InfixExpr {
            span: start_loc.until(self.prev_tok.span.end),
            lhs,
            op,
            rhs,
        })
    }

    fn assign_expr(
        &mut self,
        start_loc: Location,
        lhs: Expression<'src>,
        op: AssignOp,
    ) -> Result<AssignExpr<'src>> {
        let assignee = match lhs {
            Expression::Ident(item) => item,
            _ => {
                self.errors.push(Error::new(
                    ErrorKind::Syntax,
                    "left hand side of assignment must be an identifier".to_string(),
                    self.curr_tok.span,
                ));
                AtomExpr {
                    span: Span::default(),
                    value: "",
                }
            }
        };

        let right_prec = self.curr_tok.kind.prec().1;
        self.next()?;
        let expr = self.expression(right_prec)?;

        Ok(AssignExpr {
            span: start_loc.until(self.prev_tok.span.end),
            assignee,
            op,
            expr,
        })
    }

    fn call_expr(&mut self, start_loc: Location, expr: Expression<'src>) -> Result<CallExpr<'src>> {
        // Skip opening paren
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

        if self.curr_tok.kind != TokenKind::RParen {
            self.errors.push(Error::new(
                ErrorKind::Syntax,
                "missing closing parenthesis".to_string(),
                self.curr_tok.span,
            ));
        } else {
            self.next()?;
        }

        Ok(CallExpr {
            span: start_loc.until(self.prev_tok.span.end),
            expr,
            args,
        })
    }

    fn cast_expr(&mut self, start_loc: Location, expr: Expression<'src>) -> Result<CastExpr<'src>> {
        // Skip `as` token
        self.next()?;

        let type_ = self.type_()?;

        Ok(CastExpr {
            span: start_loc.until(self.prev_tok.span.end),
            expr,
            type_,
        })
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
            Ok(self
                .next()
                .unwrap_or_else(|| TokenKind::Eof.spanned(Span::default())))
        }
    }

    fn expr_test(
        tokens: impl IntoIterator<Item = Token<'static>>,
        tree: Expression<'static>,
    ) -> Result<()> {
        let mut parser = Parser::new(tokens.into_iter());
        parser.next()?;
        assert_eq!(parser.expression(0)?, tree);
        Ok(())
    }

    #[test]
    fn arithmetic_expressions() -> Result<()> {
        // 3-2
        expr_test(
            [
                TokenKind::Int(3).spanned(span!(0..1)),
                TokenKind::Minus.spanned(span!(1..2)),
                TokenKind::Int(2).spanned(span!(2..3)),
            ],
            Expression::Infix(Box::new(InfixExpr {
                span: span!(0..3),
                lhs: Expression::Int(AtomExpr {
                    span: span!(0..1),
                    value: 3,
                }),
                op: InfixOp::Minus,
                rhs: Expression::Int(AtomExpr {
                    span: span!(2..3),
                    value: 2,
                }),
            })),
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
            Expression::Infix(Box::new(InfixExpr {
                span: span!(0..5),
                lhs: Expression::Int(AtomExpr {
                    span: span!(0..1),
                    value: 1,
                }),
                op: InfixOp::Plus,
                rhs: Expression::Infix(Box::new(InfixExpr {
                    span: span!(2..5),
                    lhs: Expression::Int(AtomExpr {
                        span: span!(2..3),
                        value: 2,
                    }),
                    op: InfixOp::Mul,
                    rhs: Expression::Int(AtomExpr {
                        span: span!(4..5),
                        value: 3,
                    }),
                })),
            })),
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
            Expression::Infix(Box::new(InfixExpr {
                span: span!(0..7),
                lhs: Expression::Int(AtomExpr {
                    span: span!(0..1),
                    value: 2,
                }),
                op: InfixOp::Pow,
                rhs: Expression::Infix(Box::new(InfixExpr {
                    span: span!(3..7),
                    lhs: Expression::Int(AtomExpr {
                        span: span!(3..4),
                        value: 3,
                    }),
                    op: InfixOp::Pow,
                    rhs: Expression::Int(AtomExpr {
                        span: span!(6..7),
                        value: 4,
                    }),
                })),
            })),
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
            Expression::Assign(Box::new(AssignExpr {
                span: span!(0..3),
                assignee: AtomExpr {
                    span: span!(0..1),
                    value: "a",
                },
                op: AssignOp::Basic,
                expr: Expression::Int(AtomExpr {
                    span: span!(2..3),
                    value: 1,
                }),
            })),
        )?;

        // answer += 42.0
        expr_test(
            [
                TokenKind::Ident("answer").spanned(span!(0..6)),
                TokenKind::PlusAssign.spanned(span!(7..9)),
                TokenKind::Float(42.0).spanned(span!(10..14)),
            ],
            Expression::Assign(Box::new(AssignExpr {
                span: span!(0..14),
                assignee: AtomExpr {
                    span: span!(0..6),
                    value: "answer",
                },
                op: AssignOp::Plus,
                expr: Expression::Float(AtomExpr {
                    span: span!(10..14),
                    value: 42.0,
                }),
            })),
        )?;

        Ok(())
    }

    // TODO: more tests
}

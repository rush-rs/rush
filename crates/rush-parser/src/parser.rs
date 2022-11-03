use std::mem;

use crate::{ast::*, Error, ErrorKind, Lexer, Result, Span, Token, TokenKind};

pub struct Parser<'src> {
    lexer: Lexer<'src>,
    prev_tok: Token<'src>,
    curr_tok: Token<'src>,
    errors: Vec<Error>,
}

impl<'src> Parser<'src> {
    pub fn new(lexer: Lexer<'src>) -> Self {
        Self {
            lexer,
            // Initialize with dummy Eof tokens
            prev_tok: TokenKind::Eof.spanned(Span::default()),
            curr_tok: TokenKind::Eof.spanned(Span::default()),
            errors: vec![],
        }
    }

    pub fn errors(&self) -> &Vec<Error> {
        &self.errors
    }

    pub fn parse(mut self) -> Result<Program<'src>> {
        self.next()?;
        let program = self.program()?;

        if self.curr_tok.kind != TokenKind::Eof {
            self.errors.push(Error::new(
                ErrorKind::Syntax,
                format!("expected EOF, found {}", self.curr_tok.kind),
                self.curr_tok.span,
            ))
        }

        Ok(program)
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
        let mut lhs = match self.curr_tok.kind {
            TokenKind::LBrace => Expression::Block(self.block()?.into()),
            TokenKind::If => Expression::If(self.if_expr()?.into()),
            TokenKind::Int(num) => Expression::Int(num),
            TokenKind::Float(num) => Expression::Float(num),
            TokenKind::True => Expression::Bool(true),
            TokenKind::False => Expression::Bool(false),
            TokenKind::Ident(ident) => Expression::Ident(ident),
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
                TokenKind::Plus => Expression::Infix(self.infix_expr(lhs, InfixOp::Plus)?.into()),
                TokenKind::Minus => Expression::Infix(self.infix_expr(lhs, InfixOp::Minus)?.into()),
                TokenKind::Star => Expression::Infix(self.infix_expr(lhs, InfixOp::Mul)?.into()),
                TokenKind::Slash => Expression::Infix(self.infix_expr(lhs, InfixOp::Div)?.into()),
                TokenKind::Percent => Expression::Infix(self.infix_expr(lhs, InfixOp::Rem)?.into()),
                TokenKind::Pow => Expression::Infix(self.infix_expr(lhs, InfixOp::Pow)?.into()),
                TokenKind::Eq => Expression::Infix(self.infix_expr(lhs, InfixOp::Eq)?.into()),
                TokenKind::Neq => Expression::Infix(self.infix_expr(lhs, InfixOp::Neq)?.into()),
                TokenKind::Lt => Expression::Infix(self.infix_expr(lhs, InfixOp::Lt)?.into()),
                TokenKind::Gt => Expression::Infix(self.infix_expr(lhs, InfixOp::Gt)?.into()),
                TokenKind::Lte => Expression::Infix(self.infix_expr(lhs, InfixOp::Lte)?.into()),
                TokenKind::Gte => Expression::Infix(self.infix_expr(lhs, InfixOp::Gte)?.into()),
                TokenKind::Shl => Expression::Infix(self.infix_expr(lhs, InfixOp::Shl)?.into()),
                TokenKind::Shr => Expression::Infix(self.infix_expr(lhs, InfixOp::Shr)?.into()),
                TokenKind::BitOr => Expression::Infix(self.infix_expr(lhs, InfixOp::BitOr)?.into()),
                TokenKind::BitAnd => {
                    Expression::Infix(self.infix_expr(lhs, InfixOp::BitAnd)?.into())
                }
                TokenKind::BitXor => {
                    Expression::Infix(self.infix_expr(lhs, InfixOp::BitXor)?.into())
                }
                TokenKind::And => Expression::Infix(self.infix_expr(lhs, InfixOp::And)?.into()),
                TokenKind::Or => Expression::Infix(self.infix_expr(lhs, InfixOp::Or)?.into()),
                TokenKind::Assign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::Basic)?.into())
                }
                TokenKind::PlusAssign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::Plus)?.into())
                }
                TokenKind::MinusAssign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::Minus)?.into())
                }
                TokenKind::MulAssign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::Mul)?.into())
                }
                TokenKind::DivAssign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::Div)?.into())
                }
                TokenKind::RemAssign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::Rem)?.into())
                }
                TokenKind::PowAssign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::Pow)?.into())
                }
                TokenKind::ShlAssign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::Shl)?.into())
                }
                TokenKind::ShrAssign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::Shr)?.into())
                }
                TokenKind::BitOrAssign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::BitOr)?.into())
                }
                TokenKind::BitAndAssign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::BitAnd)?.into())
                }
                TokenKind::BitXorAssign => {
                    Expression::Assign(self.assign_expr(lhs, AssignOp::BitXor)?.into())
                }
                // TODO: CastExpr
                // TokenKind::As => Expression::Cast(self.cast_expr(lhs)?.into()),
                TokenKind::LParen => Expression::Call(self.call_expr(lhs)?.into()),
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

    fn prefix_expr(&mut self, op: PrefixOp) -> Result<PrefixExpr<'src>> {
        let start_loc = self.curr_tok.span.start;

        // Skip the operator token
        self.next()?;

        // PrefixExpr precedence is 27, higher than all InfixExpr precedences
        let expr = self.expression(27)?;
        Ok(PrefixExpr {
            span: start_loc.until(self.prev_tok.span.end),
            op,
            expr,
        })
    }

    fn infix_expr(&mut self, lhs: Expression<'src>, op: InfixOp) -> Result<InfixExpr<'src>> {
        let start_loc = self.curr_tok.span.start;

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

    fn assign_expr(&mut self, lhs: Expression<'src>, op: AssignOp) -> Result<AssignExpr<'src>> {
        let start_loc = self.curr_tok.span.start;

        let assignee = match lhs {
            Expression::Ident(ident) => ident,
            _ => {
                self.errors.push(Error::new(
                    ErrorKind::Syntax,
                    "left hand side of assignment must be an identifier".to_string(),
                    self.curr_tok.span,
                ));
                ""
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

    fn cast_expr(&mut self, expr: Expression<'src>) -> Result<CastExpr<'src>> {
        todo!()
    }

    fn call_expr(&mut self, expr: Expression<'src>) -> Result<CallExpr<'src>> {
        let start_loc = self.curr_tok.span.start;

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
}

use std::mem;

use either::Either;

use crate::{ast::*, Error, Lex, Location, Result, Span, Token, TokenKind};

pub struct Parser<'src, Lexer: Lex<'src>> {
    lexer: Lexer,
    prev_tok: Token<'src>,
    curr_tok: Token<'src>,
    errors: Vec<Error<'src>>,
}

impl<'src, Lexer: Lex<'src>> Parser<'src, Lexer> {
    /// Creates a new Parser
    pub fn new(lexer: Lexer) -> Self {
        Self {
            lexer,
            // initialize with dummy Eof tokens
            prev_tok: Token::dummy(),
            curr_tok: Token::dummy(),
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
    pub fn parse(mut self) -> (Result<'src, Program<'src>>, Vec<Error<'src>>) {
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
                self.lexer.source(),
            ))
        }

        (Ok(program), self.errors)
    }

    // moves cursor to next token
    fn next(&mut self) -> Result<'src, ()> {
        // swap prev_tok and curr_tok in memory so that what was curr_tok is now prev_tok
        mem::swap(&mut self.prev_tok, &mut self.curr_tok);
        // overwrite curr_tok (which is now what prev_tok was) with the next token from the lexer
        self.curr_tok = self.lexer.next_token()?;

        Ok(())
    }

    // expects the curr_tok to be of the specified kind
    fn expect(&mut self, kind: TokenKind) -> Result<'src, ()> {
        if self.curr_tok.kind != kind {
            return Err(Error::new_boxed(
                format!("expected `{kind}`, found `{}`", self.curr_tok.kind),
                self.curr_tok.span,
                self.lexer.source(),
            ));
        }
        self.next()?;
        Ok(())
    }

    // expects the curr_tok to be an identifier and returns its name if this is the case
    fn expect_ident(&mut self) -> Result<'src, Spanned<'src, &'src str>> {
        match self.curr_tok.kind {
            TokenKind::Ident(ident) => {
                let ident = Spanned {
                    span: self.curr_tok.span,
                    inner: ident,
                };
                self.next()?;
                Ok(ident)
            }
            _ => Err(Error::new_boxed(
                format!("expected identifier, found `{}`", self.curr_tok.kind),
                self.curr_tok.span,
                self.lexer.source(),
            )),
        }
    }

    // expects curr_tok to be the specified token kind and adds a soft error otherwise
    fn expect_recoverable(
        &mut self,
        kind: TokenKind,
        message: impl Into<String>,
        span: Span<'src>,
    ) -> Result<'src, Span<'src>> {
        let start_loc = self.curr_tok.span.start;
        let end_loc = if self.curr_tok.kind != kind {
            self.errors
                .push(Error::new(message.into(), span, self.lexer.source()));
            self.curr_tok.span.start
        } else {
            self.next()?;
            self.prev_tok.span.end
        };
        Ok(start_loc.until(end_loc))
    }

    //////////////////////////

    fn program(&mut self) -> Result<'src, Program<'src>> {
        let start_loc = self.curr_tok.span.start;
        let mut functions = vec![];
        let mut globals = vec![];

        while self.curr_tok.kind != TokenKind::Eof {
            match self.curr_tok.kind {
                TokenKind::Fn => functions.push(self.function_definition()?),
                TokenKind::Let => globals.push(self.let_stmt()?),
                _ => {
                    return Err(Error::new_boxed(
                        format!(
                            "expected either `fn` or `let`, found `{}`",
                            self.curr_tok.kind
                        ),
                        self.curr_tok.span,
                        self.lexer.source(),
                    ))
                }
            }
        }

        Ok(Program {
            span: start_loc.until(self.prev_tok.span.end),
            functions,
            globals,
        })
    }

    fn type_(&mut self) -> Result<'src, Spanned<'src, Type>> {
        let start_loc = self.curr_tok.span.start;
        let type_ = match self.curr_tok.kind {
            TokenKind::Ident("int") => Type::Int,
            TokenKind::Ident("float") => Type::Float,
            TokenKind::Ident("bool") => Type::Bool,
            TokenKind::Ident("char") => Type::Char,
            TokenKind::Ident(ident) => {
                self.errors.push(Error::new(
                    format!("unknown type `{ident}`"),
                    self.curr_tok.span,
                    self.lexer.source(),
                ));
                Type::Unknown
            }
            TokenKind::LParen => {
                self.next()?;
                self.expect_recoverable(
                    TokenKind::RParen,
                    "missing closing parenthesis",
                    self.curr_tok.span,
                )?;
                return Ok(Spanned {
                    span: start_loc.until(self.prev_tok.span.end),
                    inner: Type::Unit,
                });
            }
            invalid => {
                return Err(Error::new_boxed(
                    format!("expected a type, found `{invalid}`"),
                    self.curr_tok.span,
                    self.lexer.source(),
                ));
            }
        };
        self.next()?;
        Ok(Spanned {
            span: start_loc.until(self.prev_tok.span.end),
            inner: type_,
        })
    }

    fn function_definition(&mut self) -> Result<'src, FunctionDefinition<'src>> {
        let start_loc = self.curr_tok.span.start;

        self.expect(TokenKind::Fn)?;
        let name = self.expect_ident()?;
        let l_paren = self.curr_tok.span;
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

        let r_paren = self.expect_recoverable(
            TokenKind::RParen,
            "missing closing parenthesis",
            self.curr_tok.span,
        )?;

        let params = Spanned {
            span: l_paren.start.until(r_paren.end),
            inner: params,
        };

        let return_type = match self.curr_tok.kind {
            TokenKind::Arrow => {
                self.next()?;
                let type_ = self.type_()?;
                Spanned {
                    span: type_.span,
                    inner: Some(type_.inner),
                }
            }
            _ => Spanned {
                span: r_paren.start.until(self.curr_tok.span.end),
                inner: None,
            },
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

    fn parameter(&mut self) -> Result<'src, Parameter<'src>> {
        let mutable = self.curr_tok.kind == TokenKind::Mut;
        if mutable {
            self.next()?;
        }

        let name = self.expect_ident()?;
        let type_ = match self.curr_tok.kind {
            TokenKind::Comma | TokenKind::RParen => {
                self.errors.push(Error::new(
                    format!("missing type for parameter `{}`", name.inner),
                    name.span,
                    self.lexer.source(),
                ));
                Spanned {
                    span: Span::dummy(),
                    inner: Type::Unknown,
                }
            }
            _ => {
                self.expect(TokenKind::Colon)?;
                self.type_()?
            }
        };
        Ok(Parameter {
            mutable,
            name,
            type_,
        })
    }

    fn block(&mut self) -> Result<'src, Block<'src>> {
        let start_loc = self.curr_tok.span.start;

        self.expect(TokenKind::LBrace)?;

        let mut stmts = vec![];
        let mut expr = None;
        while !matches!(self.curr_tok.kind, TokenKind::RBrace | TokenKind::Eof) {
            match self.statement()? {
                Either::Left(stmt) => stmts.push(stmt),
                Either::Right(expression) => {
                    expr = Some(expression);
                    break;
                }
            }
        }

        self.expect_recoverable(
            TokenKind::RBrace,
            "missing closing brace",
            self.curr_tok.span,
        )?;

        Ok(Block {
            span: start_loc.until(self.prev_tok.span.end),
            stmts,
            expr,
        })
    }

    fn statement(&mut self) -> Result<'src, Either<Statement<'src>, Expression<'src>>> {
        Ok(match self.curr_tok.kind {
            TokenKind::Let => Either::Left(Statement::Let(self.let_stmt()?)),
            TokenKind::Return => Either::Left(self.return_stmt()?),
            TokenKind::Loop => Either::Left(self.loop_stmt()?),
            TokenKind::While => Either::Left(self.while_stmt()?),
            TokenKind::For => Either::Left(self.for_stmt()?),
            TokenKind::Break => Either::Left(self.break_stmt()?),
            TokenKind::Continue => Either::Left(self.continue_stmt()?),
            _ => self.expr_stmt()?,
        })
    }

    fn let_stmt(&mut self) -> Result<'src, LetStmt<'src>> {
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
        self.expect_recoverable(
            TokenKind::Semicolon,
            "missing semicolon after statement",
            start_loc.until(self.prev_tok.span.end),
        )?;

        Ok(LetStmt {
            span: start_loc.until(self.prev_tok.span.end),
            mutable,
            type_,
            name,
            expr,
        })
    }

    fn return_stmt(&mut self) -> Result<'src, Statement<'src>> {
        let start_loc = self.curr_tok.span.start;

        // skip return token: this function is only called when self.curr_tok.kind == TokenKind::Return
        self.next()?;

        let expr = match self.curr_tok.kind {
            TokenKind::Semicolon | TokenKind::RBrace | TokenKind::Eof => None,
            _ => Some(self.expression(0)?),
        };

        self.expect_recoverable(
            TokenKind::Semicolon,
            "missing semicolon after statement",
            start_loc.until(self.prev_tok.span.end),
        )?;

        Ok(Statement::Return(ReturnStmt {
            span: start_loc.until(self.prev_tok.span.end),
            expr,
        }))
    }

    fn loop_stmt(&mut self) -> Result<'src, Statement<'src>> {
        let start_loc = self.curr_tok.span.start;

        // skip loop token: this function is only called when self.curr_tok.kind == TokenKind::Loop
        self.next()?;

        let block = self.block()?;

        // skip optional semicolon
        if self.curr_tok.kind == TokenKind::Semicolon {
            self.next()?;
        }

        Ok(Statement::Loop(LoopStmt {
            span: start_loc.until(self.prev_tok.span.end),
            block,
        }))
    }

    fn while_stmt(&mut self) -> Result<'src, Statement<'src>> {
        let start_loc = self.curr_tok.span.start;

        // skip while token: this function is only called when self.curr_tok.kind == TokenKind::While
        self.next()?;

        let cond = self.expression(0)?;

        let block = self.block()?;

        // skip optional semicolon
        if self.curr_tok.kind == TokenKind::Semicolon {
            self.next()?;
        }

        Ok(Statement::While(WhileStmt {
            span: start_loc.until(self.prev_tok.span.end),
            cond,
            block,
        }))
    }

    fn for_stmt(&mut self) -> Result<'src, Statement<'src>> {
        let start_loc = self.curr_tok.span.start;

        // skip for token: this function is only called when self.curr_tok.kind == TokenKind::For
        self.next()?;

        // parse the initializer
        let ident = self.expect_ident()?;
        self.expect_recoverable(
            TokenKind::Assign,
            format!("expected `=`, found `{}`", self.curr_tok.kind),
            self.curr_tok.span,
        )?;
        let initializer = self.expression(0)?;

        // parse the condition expression
        self.expect_recoverable(
            TokenKind::Semicolon,
            format!("expected semicolon, found `{}`", self.curr_tok.kind),
            self.curr_tok.span,
        )?;
        let cond = self.expression(0)?;

        // parse the update expression
        self.expect_recoverable(
            TokenKind::Semicolon,
            format!("expected semicolon, found `{}`", self.curr_tok.kind),
            self.curr_tok.span,
        )?;
        let update = self.expression(0)?;

        // parse the block
        let block = self.block()?;

        // skip optional semicolon
        if self.curr_tok.kind == TokenKind::Semicolon {
            self.next()?;
        }

        Ok(Statement::For(ForStmt {
            span: start_loc.until(self.prev_tok.span.end),
            ident,
            initializer,
            cond,
            update,
            block,
        }))
    }

    fn break_stmt(&mut self) -> Result<'src, Statement<'src>> {
        let start_loc = self.curr_tok.span.start;

        // skip break token: this function is only called when self.curr_tok.kind == TokenKind::Break
        self.next()?;

        self.expect_recoverable(
            TokenKind::Semicolon,
            "missing semicolon after statement",
            self.prev_tok.span,
        )?;

        Ok(Statement::Break(BreakStmt {
            span: start_loc.until(self.prev_tok.span.end),
        }))
    }

    fn continue_stmt(&mut self) -> Result<'src, Statement<'src>> {
        let start_loc = self.curr_tok.span.start;

        // skip continue token: this function is only called when self.curr_tok.kind == TokenKind::Continue
        self.next()?;

        self.expect_recoverable(
            TokenKind::Semicolon,
            "missing semicolon after statement",
            self.prev_tok.span,
        )?;

        Ok(Statement::Continue(ContinueStmt {
            span: start_loc.until(self.prev_tok.span.end),
        }))
    }

    fn expr_stmt(&mut self) -> Result<'src, Either<Statement<'src>, Expression<'src>>> {
        let start_loc = self.curr_tok.span.start;

        let (expr, with_block) = match self.curr_tok.kind {
            TokenKind::If => (Expression::If(self.if_expr()?.into()), true),
            TokenKind::LBrace => (Expression::Block(self.block()?.into()), true),
            _ => (self.expression(0)?, false),
        };

        match (self.curr_tok.kind, with_block) {
            (TokenKind::Semicolon, true) => self.next()?,
            (TokenKind::Semicolon, false) => self.next()?,
            (TokenKind::RBrace, _) => return Ok(Either::Right(expr)),
            (_, true) => {}
            (_, false) => self.errors.push(Error::new(
                "missing semicolon after statement".to_string(),
                expr.span(),
                self.lexer.source(),
            )),
        }

        Ok(Either::Left(Statement::Expr(ExprStmt {
            span: start_loc.until(self.prev_tok.span.end),
            expr,
        })))
    }

    fn expression(&mut self, prec: u8) -> Result<'src, Expression<'src>> {
        let start_loc = self.curr_tok.span.start;

        let mut lhs = match self.curr_tok.kind {
            TokenKind::LBrace => Expression::Block(self.block()?.into()),
            TokenKind::If => Expression::If(self.if_expr()?.into()),
            TokenKind::Int(num) => Expression::Int(self.atom(num)?),
            TokenKind::Float(num) => Expression::Float(self.atom(num)?),
            TokenKind::True => Expression::Bool(self.atom(true)?),
            TokenKind::False => Expression::Bool(self.atom(false)?),
            TokenKind::Char(char) => Expression::Char(self.atom(char)?),
            TokenKind::Ident(ident) => Expression::Ident(self.atom(ident)?),
            TokenKind::Not => Expression::Prefix(self.prefix_expr(PrefixOp::Not)?.into()),
            TokenKind::Minus => Expression::Prefix(self.prefix_expr(PrefixOp::Neg)?.into()),
            TokenKind::LParen => Expression::Grouped(self.grouped_expr()?),
            invalid => {
                return Err(Error::new_boxed(
                    format!("expected an expression, found `{invalid}`"),
                    self.curr_tok.span,
                    self.lexer.source(),
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

    fn if_expr(&mut self) -> Result<'src, IfExpr<'src>> {
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
                            stmts: vec![],
                            expr: Some(Expression::If(if_expr.into())),
                        }
                    }
                    TokenKind::LBrace => self.block()?,
                    invalid => {
                        return Err(Error::new_boxed(
                            format!(
                                "expected either `if` or block after `else`, found `{invalid}`"
                            ),
                            self.curr_tok.span,
                            self.lexer.source(),
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

    fn atom<T>(&mut self, inner: T) -> Result<'src, Spanned<'src, T>> {
        let start_loc = self.curr_tok.span.start;
        self.next()?;
        Ok(Spanned {
            span: start_loc.until(self.prev_tok.span.end),
            inner,
        })
    }

    fn prefix_expr(&mut self, op: PrefixOp) -> Result<'src, PrefixExpr<'src>> {
        let start_loc = self.curr_tok.span.start;

        // skip the operator token
        self.next()?;

        // PrefixExpr precedence is 27, higher than all InfixExpr precedences except CallExpr
        let expr = self.expression(27)?;
        Ok(PrefixExpr {
            span: start_loc.until(self.prev_tok.span.end),
            op,
            expr,
        })
    }

    fn grouped_expr(&mut self) -> Result<'src, Spanned<'src, Box<Expression<'src>>>> {
        let start_loc = self.curr_tok.span.start;
        // skip the opening parenthesis
        self.next()?;

        let expr = self.expression(0)?;
        self.expect_recoverable(
            TokenKind::RParen,
            "missing closing parenthesis",
            self.curr_tok.span,
        )?;

        Ok(Spanned {
            span: start_loc.until(self.prev_tok.span.end),
            inner: expr.into(),
        })
    }

    fn infix_expr(
        &mut self,
        start_loc: Location<'src>,
        lhs: Expression<'src>,
        op: InfixOp,
    ) -> Result<'src, Expression<'src>> {
        let right_prec = self.curr_tok.kind.prec().1;
        self.next()?;
        let rhs = self.expression(right_prec)?;

        Ok(Expression::Infix(
            InfixExpr {
                span: start_loc.until(self.prev_tok.span.end),
                lhs,
                op,
                rhs,
            }
            .into(),
        ))
    }

    fn assign_expr(
        &mut self,
        start_loc: Location<'src>,
        lhs: Expression<'src>,
        op: AssignOp,
    ) -> Result<'src, Expression<'src>> {
        let assignee = match lhs {
            Expression::Ident(item) => item,
            _ => {
                self.errors.push(Error::new(
                    "left hand side of assignment must be an identifier".to_string(),
                    self.prev_tok.span,
                    self.lexer.source(),
                ));
                Spanned {
                    span: Span::dummy(),
                    inner: "",
                }
            }
        };

        let right_prec = self.curr_tok.kind.prec().1;
        self.next()?;
        let expr = self.expression(right_prec)?;

        Ok(Expression::Assign(
            AssignExpr {
                span: start_loc.until(self.prev_tok.span.end),
                assignee,
                op,
                expr,
            }
            .into(),
        ))
    }

    fn call_expr(
        &mut self,
        start_loc: Location<'src>,
        expr: Expression<'src>,
    ) -> Result<'src, Expression<'src>> {
        let func = match expr {
            Expression::Ident(func) => func,
            _ => {
                self.errors.push(Error::new(
                    "only identifiers can be called".to_string(),
                    self.curr_tok.span,
                    self.lexer.source(),
                ));
                Spanned {
                    span: Span::dummy(),
                    inner: "",
                }
            }
        };

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

        self.expect_recoverable(
            TokenKind::RParen,
            "missing closing parenthesis",
            self.curr_tok.span,
        )?;

        Ok(Expression::Call(
            CallExpr {
                span: start_loc.until(self.prev_tok.span.end),
                func,
                args,
            }
            .into(),
        ))
    }

    fn cast_expr(
        &mut self,
        start_loc: Location<'src>,
        expr: Expression<'src>,
    ) -> Result<'src, Expression<'src>> {
        // skip `as` token
        self.next()?;

        let type_ = self.type_()?;

        Ok(Expression::Cast(
            CastExpr {
                span: start_loc.until(self.prev_tok.span.end),
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
        fn next_token(&mut self) -> Result<'src, Token<'src>> {
            Ok(self.next().unwrap_or_else(Token::dummy))
        }

        fn source(&self) -> &'src str {
            ""
        }
    }

    /// Parses the `tokens` into an [`Expression`] and asserts equality with `tree`
    /// without any errors
    fn expr_test(
        tokens: impl IntoIterator<Item = Token<'static>>,
        tree: Expression<'static>,
    ) -> Result<'static, ()> {
        let mut parser = Parser::new(tokens.into_iter());
        parser.next()?;
        let expr = parser.expression(0)?;
        assert!(dbg!(parser.errors).is_empty());
        assert_eq!(dbg!(expr), dbg!(tree));
        Ok(())
    }

    /// Parses the `tokens` into a [`Statement`] and asserts equality with `tree`
    /// without any errors
    fn stmt_test(
        tokens: impl IntoIterator<Item = Token<'static>>,
        tree: Statement<'static>,
    ) -> Result<'static, ()> {
        let mut parser = Parser::new(tokens.into_iter());
        parser.next()?;
        let stmt = parser.statement()?;
        assert!(dbg!(parser.errors).is_empty());
        assert_eq!(dbg!(stmt), Either::Left(dbg!(tree)));
        Ok(())
    }

    /// Parses the `tokens` into a [`Program`] and asserts equality with `tree`
    /// without any errors
    fn program_test(
        tokens: impl IntoIterator<Item = Token<'static>>,
        tree: Program<'static>,
    ) -> Result<'static, ()> {
        let parser = Parser::new(tokens.into_iter());
        let (res, errors) = parser.parse();
        assert!(dbg!(errors).is_empty());
        assert_eq!(dbg!(res?), dbg!(tree));
        Ok(())
    }

    #[test]
    fn arithmetic_expressions() -> Result<'static, ()> {
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
                        lhs: (Int 3, @ 0..1),
                        op: InfixOp::Minus,
                        rhs: (Int 2, @ 2..3)),
                    op: InfixOp::Minus,
                    rhs: (Int 1, @ 4..5))
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
                    lhs: (Int 1, @ 0..1),
                    op: InfixOp::Plus,
                    rhs: (InfixExpr @ 2..5,
                        lhs: (Int 2, @ 2..3),
                        op: InfixOp::Mul,
                        rhs: (Int 3, @ 4..5)))
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
                    lhs: (Int 2, @ 0..1),
                    op: InfixOp::Pow,
                    rhs: (InfixExpr @ 3..7,
                        lhs: (Int 3, @ 3..4),
                        op: InfixOp::Pow,
                        rhs: (Int 4, @ 6..7)))
            },
        )?;

        Ok(())
    }

    #[test]
    fn assignment_expressions() -> Result<'static, ()> {
        // a=1
        expr_test(
            tokens![
                Ident("a") @ 0..1,
                Assign @ 1..2,
                Int(1) @ 2..3,
            ],
            tree! {
                (AssignExpr @ 0..3,
                    assignee: ("a", @ 0..1),
                    op: AssignOp::Basic,
                    expr: (Int 1, @ 2..3))
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
                    assignee: ("answer", @ 0..6),
                    op: AssignOp::Plus,
                    expr: (InfixExpr @ 10..19,
                        lhs: (Float 42.0, @ 10..14),
                        op: InfixOp::Minus,
                        rhs: (Float 0.0, @ 17..19)))
            },
        )?;

        Ok(())
    }

    #[test]
    fn if_expr() -> Result<'static, ()> {
        // if true{}
        expr_test(
            tokens![
                If @ 0..2,
                True @ 3..7,
                LBrace @ 7..8,
                RBrace @ 8..9,
            ],
            tree! {
                (IfExpr @ 0..9,
                    cond: (Bool true, @ 3..7),
                    then_block: (Block @ 7..9,
                        stmts: [],
                        expr: (None)),
                    else_block: (None))
            },
        )?;

        // if 2 > 1 { 1 } else { { 2 } }
        expr_test(
            tokens![
                If @ 0..2,
                Int(2) @ 3..4,
                Gt @ 5..6,
                Int(1) @ 7..8,
                LBrace @ 9..10,
                Int(1) @ 11..12,
                RBrace @ 13..14,
                Else @ 15..19,
                LBrace @ 20..21,
                LBrace @ 22..23,
                Int(2) @ 24..25,
                RBrace @ 26..27,
                RBrace @ 28..29,
            ],
            tree! {
                (IfExpr @ 0..29,
                    cond: (InfixExpr @ 3..8,
                        lhs: (Int 2, @ 3..4),
                        op: InfixOp::Gt,
                        rhs: (Int 1, @ 7..8)),
                    then_block: (Block @ 9..14,
                        stmts: [],
                        expr: (Some(Int 1, @ 11..12))),
                    else_block: (Some(Block @ 20..29,
                        stmts: [],
                        expr: (Some(BlockExpr @ 22..27,
                            stmts: [],
                            expr: (Some(Int 2, @ 24..25)))))))
            },
        )?;

        // if false {} else if cond {1;}
        expr_test(
            tokens![
                If @ 0..2,
                False @ 3..8,
                LBrace @ 9..10,
                RBrace @ 10..11,
                Else @ 12..16,
                If @ 17..19,
                Ident("cond") @ 20..24,
                LBrace @ 25..26,
                Int(1) @ 26..27,
                Semicolon @ 27..28,
                RBrace @ 28..29,
            ],
            tree! {
                (IfExpr @ 0..29,
                    cond: (Bool false, @ 3..8),
                    then_block: (Block @ 9..11,
                        stmts: [],
                        expr: (None)),
                    else_block: (Some(Block @ 17..29,
                        stmts: [],
                        expr: (Some(IfExpr @ 17..29,
                            cond: (Ident "cond", @ 20..24),
                            then_block: (Block @ 25..29,
                                stmts: [
                                    (ExprStmt @ 26..28, (Int 1, @ 26..27))],
                                expr: (None)),
                            else_block: (None))))))
            },
        )?;

        Ok(())
    }

    #[test]
    fn prefix_expressions() -> Result<'static, ()> {
        // !func()
        expr_test(
            tokens![
                Not @ 0..1,
                Ident("func") @ 1..5,
                LParen @ 5..6,
                RParen @ 6..7,
            ],
            tree! {
                (PrefixExpr @ 0..7,
                    op: PrefixOp::Not,
                    expr: (CallExpr @ 1..7,
                        func: ("func", @ 1..5),
                        args: []))
            },
        )?;

        Ok(())
    }

    fn infix_expr_test(token: TokenKind<'static>, op: InfixOp) -> Result<()> {
        expr_test(
            [
                TokenKind::Int(1).spanned(span!(0..1)),
                token.spanned(span!(1..2)),
                TokenKind::Int(2).spanned(span!(2..3)),
            ],
            tree! {
                (InfixExpr @ 0..3,
                    lhs: (Int 1, @ 0..1),
                    op: op,
                    rhs: (Int 2, @ 2..3))
            },
        )
    }

    #[test]
    fn infix_expr() -> Result<'static, ()> {
        infix_expr_test(TokenKind::Plus, InfixOp::Plus)?;
        infix_expr_test(TokenKind::Minus, InfixOp::Minus)?;
        infix_expr_test(TokenKind::Star, InfixOp::Mul)?;
        infix_expr_test(TokenKind::Slash, InfixOp::Div)?;
        infix_expr_test(TokenKind::Percent, InfixOp::Rem)?;
        infix_expr_test(TokenKind::Pow, InfixOp::Pow)?;
        infix_expr_test(TokenKind::Eq, InfixOp::Eq)?;
        infix_expr_test(TokenKind::Neq, InfixOp::Neq)?;
        infix_expr_test(TokenKind::Lt, InfixOp::Lt)?;
        infix_expr_test(TokenKind::Gt, InfixOp::Gt)?;
        infix_expr_test(TokenKind::Lte, InfixOp::Lte)?;
        infix_expr_test(TokenKind::Gte, InfixOp::Gte)?;
        infix_expr_test(TokenKind::Shl, InfixOp::Shl)?;
        infix_expr_test(TokenKind::Shr, InfixOp::Shr)?;
        infix_expr_test(TokenKind::BitOr, InfixOp::BitOr)?;
        infix_expr_test(TokenKind::BitAnd, InfixOp::BitAnd)?;
        infix_expr_test(TokenKind::BitXor, InfixOp::BitXor)?;
        infix_expr_test(TokenKind::And, InfixOp::And)?;
        infix_expr_test(TokenKind::Or, InfixOp::Or)?;

        Ok(())
    }

    fn assign_expr_test(token: TokenKind<'static>, op: AssignOp) -> Result<()> {
        expr_test(
            [
                TokenKind::Ident("a").spanned(span!(0..1)),
                token.spanned(span!(1..2)),
                TokenKind::Int(2).spanned(span!(2..3)),
            ],
            tree! {
                (AssignExpr @ 0..3,
                    assignee: ("a", @ 0..1),
                    op: op,
                    expr: (Int 2, @ 2..3))
            },
        )
    }

    #[test]
    fn assign_expr() -> Result<'static, ()> {
        assign_expr_test(TokenKind::Assign, AssignOp::Basic)?;
        assign_expr_test(TokenKind::PlusAssign, AssignOp::Plus)?;
        assign_expr_test(TokenKind::MinusAssign, AssignOp::Minus)?;
        assign_expr_test(TokenKind::MulAssign, AssignOp::Mul)?;
        assign_expr_test(TokenKind::DivAssign, AssignOp::Div)?;
        assign_expr_test(TokenKind::RemAssign, AssignOp::Rem)?;
        assign_expr_test(TokenKind::PowAssign, AssignOp::Pow)?;
        assign_expr_test(TokenKind::ShlAssign, AssignOp::Shl)?;
        assign_expr_test(TokenKind::ShrAssign, AssignOp::Shr)?;
        assign_expr_test(TokenKind::BitOrAssign, AssignOp::BitOr)?;
        assign_expr_test(TokenKind::BitAndAssign, AssignOp::BitAnd)?;
        assign_expr_test(TokenKind::BitXorAssign, AssignOp::BitXor)?;

        Ok(())
    }

    #[test]
    fn let_stmt() -> Result<'static, ()> {
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
                    name: ("a", @ 4..5),
                    type: (None),
                    expr: (Int 1, @ 6..7))
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
                    name: ("b", @ 8..9),
                    type: (None),
                    expr: (Int 2, @ 12..13))
            },
        )?;

        // let c: float = 3f;
        stmt_test(
            tokens![
                Let @ 0..3,
                Ident("c") @ 4..5,
                Colon @ 5..6,
                Ident("float") @ 7..12,
                Assign @ 13..14,
                Float(3.0) @ 15..17,
                Semicolon @ 17..18,
            ],
            tree! {
                (LetStmt @ 0..18,
                    mutable: false,
                    name: ("c", @ 4..5),
                    type: (Some(Type::Float, @ 7..12)),
                    expr: (Float 3.0, @ 15..17))
            },
        )?;

        Ok(())
    }

    #[test]
    fn return_stmt() -> Result<'static, ()> {
        // return;
        stmt_test(
            tokens![
                Return @ 0..6,
                Semicolon @ 6..7,
            ],
            tree! { (ReturnStmt @ 0..7, (None)) },
        )?;

        // return 1;
        stmt_test(
            tokens![
                Return @ 0..6,
                Int(1) @ 6..7,
                Semicolon @ 7..8,
            ],
            tree! { (ReturnStmt @ 0..8, (Some(Int 1, @ 6..7))) },
        )?;

        Ok(())
    }

    #[test]
    fn expr_stmt() -> Result<'static, ()> {
        // 1;
        stmt_test(
            tokens![
                Int(1) @ 0..1,
                Semicolon @ 1..2,
            ],
            tree! { (ExprStmt @ 0..2, (Int 1, @ 0..1)) },
        )?;

        // {}
        stmt_test(
            tokens![
                LBrace @ 0..1,
                RBrace @ 1..2,
            ],
            tree! {
                (ExprStmt @ 0..2, (BlockExpr @ 0..2,
                    stmts: [],
                    expr: (None)))
            },
        )?;

        Ok(())
    }

    #[test]
    fn programs() -> Result<'static, ()> {
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
                (Program @ 0..30,
                    functions: [
                        (FunctionDefinition @ 0..30,
                            name: ("main", @ 3..7),
                            params @ 7..9: [],
                            return_type: (None, @ 8..11),
                            block: (Block @ 10..30,
                                stmts: [
                                    (LetStmt @ 12..25,
                                        mutable: false,
                                        name: ("a", @ 16..17),
                                        type: (None),
                                        expr: (Bool true, @ 20..24)),
                                    (ExprStmt @ 26..28, (Ident "a", @ 26..27))],
                                expr: (None)))],
                    globals: [])
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
                (Program @ 0..61,
                    functions: [
                        (FunctionDefinition @ 0..61,
                            name: ("add", @ 3..6),
                            params @ 6..29: [
                                (Parameter,
                                    mutable: false,
                                    name: ("left", @ 7..11),
                                    type: (Type::Int, @ 13..16)),
                                (Parameter,
                                    mutable: false,
                                    name: ("right", @ 18..23),
                                    type: (Type::Int, @ 25..28))],
                            return_type: (Some(Type::Int), @ 33..36),
                            block: (Block @ 37..61,
                                stmts: [
                                    (ReturnStmt @ 39..59, (Some(InfixExpr @ 46..58,
                                        lhs: (Ident "left", @ 46..50),
                                        op: InfixOp::Plus,
                                        rhs: (Ident "right", @ 53..58))))],
                                expr: (None)))],
                    globals: [])
            },
        )?;

        // fn a() {} fn b() {} let a = 1;
        program_test(
            tokens![
                Fn @ 0..2,
                Ident("a") @ 3..4,
                LParen @ 4..5,
                RParen @ 5..6,
                LBrace @ 7..8,
                RBrace @ 8..9,
                Fn @ 10..12,
                Ident("b") @ 13..14,
                LParen @ 14..15,
                RParen @ 15..16,
                LBrace @ 17..18,
                RBrace @ 18..19,
                Let @ 20..23,
                Ident("a") @ 24..25,
                Assign @ 26..27,
                Int(1) @ 28..29,
                Semicolon @ 29..30,
            ],
            tree! {
                (Program @ 0..30,
                    functions: [
                        (FunctionDefinition @ 0..9,
                            name: ("a", @ 3..4),
                            params @ 4..6: [],
                            return_type: (None, @ 5..8),
                            block: (Block @ 7..9,
                                stmts: [],
                                expr: (None))),
                        (FunctionDefinition @ 10..19,
                            name: ("b", @ 13..14),
                            params @ 14..16: [],
                            return_type: (None, @ 15..18),
                            block: (Block @ 17..19,
                                stmts: [],
                                expr: (None)))],
                    globals: [
                        (Let @ 20..30,
                            mutable: false,
                            name: ("a", @ 24..25),
                            type: (None),
                            expr: (Int 1, @ 28..29))])
            },
        )?;

        Ok(())
    }
}

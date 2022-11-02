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
                // TODO: use Display trait
                format!("expected EOF, found {:?}", self.curr_tok.kind),
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
                // TODO: use Display trait
                format!("expected {kind:?}, found {:?}", self.curr_tok.kind),
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
                format!("expected identifier, found {:?}", self.curr_tok.kind),
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
                    // TODO: use Display trait
                    format!("expected a type, found {kind:?}"),
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
        if self.curr_tok.kind != TokenKind::RParen {
            params.push(self.parameter()?);
            while self.curr_tok.kind == TokenKind::Comma {
                self.next()?;
                if self.curr_tok.kind == TokenKind::RParen {
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
        while self.curr_tok.kind != TokenKind::RBrace && self.curr_tok.kind != TokenKind::Eof {
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
        self.expect(TokenKind::Assign)?;
        let expr = self.expression()?;

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
            _ => Some(self.expression()?),
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
            TokenKind::If => (Expression::If(Box::new(self.if_expr()?)), true),
            TokenKind::LBrace => (Expression::Block(Box::new(self.block()?)), true),
            _ => (self.expression()?, false),
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

    fn expression(&mut self) -> Result<Expression<'src>> {
        todo!()
    }

    fn if_expr(&mut self) -> Result<IfExpr<'src>> {
        todo!()
    }

    fn prefix_expr(&mut self) -> Result<PrefixExpr<'src>> {
        todo!()
    }

    fn infix_expr(&mut self) -> Result<InfixExpr<'src>> {
        todo!()
    }

    fn call_expr(&mut self) -> Result<CallExpr<'src>> {
        todo!()
    }
}

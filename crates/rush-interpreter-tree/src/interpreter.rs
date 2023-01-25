use std::{borrow::Cow, cell::RefCell, collections::HashMap, rc::Rc};

use rush_analyzer::{ast::*, AssignOp, InfixOp, PrefixOp, Type};

use crate::value::{InterruptKind, Value};

pub(crate) type Error = Cow<'static, str>;
type ExprResult = Result<Value, InterruptKind>;
type StmtResult = Result<(), InterruptKind>;
type Scope<'src> = HashMap<&'src str, Rc<RefCell<Value>>>;

#[derive(Debug, Default)]
pub struct Interpreter<'src> {
    scopes: Vec<Scope<'src>>,
    functions: HashMap<&'src str, Rc<AnalyzedFunctionDefinition<'src>>>,
}

impl<'src> Interpreter<'src> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(mut self, tree: AnalyzedProgram<'src>) -> Result<i64, Error> {
        for func in tree.functions.into_iter().filter(|f| f.used) {
            self.functions.insert(func.name, func.into());
        }

        let mut global_scope = HashMap::new();
        for global in tree.globals.iter().filter(|g| g.used) {
            global_scope.insert(
                global.name,
                match global.expr {
                    AnalyzedExpression::Int(num) => Value::Int(num).wrapped(),
                    AnalyzedExpression::Float(num) => Value::Float(num).wrapped(),
                    AnalyzedExpression::Bool(bool) => Value::Bool(bool).wrapped(),
                    AnalyzedExpression::Char(num) => Value::Char(num).wrapped(),
                    _ => unreachable!("the analyzer guarantees constant globals"),
                },
            );
        }
        self.scopes.push(global_scope);

        self.functions.insert(
            "main",
            AnalyzedFunctionDefinition {
                used: true,
                name: "main",
                params: vec![],
                return_type: Type::Unit,
                block: tree.main_fn,
            }
            .into(),
        );

        // ignore interruptions (e.g. break, return)
        match self.call_func("main", vec![]) {
            Err(InterruptKind::Error(msg)) => Err(msg),
            Err(InterruptKind::Exit(code)) => Ok(code),
            Ok(_) | Err(_) => Ok(0),
        }
    }

    //////////////////////////////////

    fn get_var(&mut self, name: &'src str) -> Rc<RefCell<Value>> {
        for scope in self.scopes.iter().rev() {
            if let Some(var) = scope.get(name) {
                return Rc::clone(var);
            }
        }
        unreachable!("the analyzer guarantees valid variable references")
    }

    fn scoped<T>(&mut self, scope: Scope<'src>, callback: impl FnOnce(&mut Self) -> T) -> T {
        self.scopes.push(scope);
        let res = callback(self);
        self.scopes.pop();
        res
    }

    fn call_func(&mut self, func_name: &'src str, mut args: Vec<Value>) -> ExprResult {
        if func_name == "exit" {
            return Err(InterruptKind::Exit(args.swap_remove(0).unwrap_int()));
        }

        let func = Rc::clone(&self.functions[func_name]);

        let mut scope = HashMap::new();
        for (param, arg) in func.params.iter().zip(args) {
            scope.insert(param.name, arg.wrapped());
        }

        self.scoped(scope, |self_| match self_.visit_block(&func.block, false) {
            Ok(val) => Ok(val),
            Err(interrupt) => Ok(interrupt.into_value()?),
        })
    }

    //////////////////////////////////

    fn visit_block(&mut self, node: &AnalyzedBlock<'src>, new_scope: bool) -> ExprResult {
        let callback = |self_: &mut Self| {
            for stmt in &node.stmts {
                self_.visit_statement(stmt)?;
            }
            node.expr
                .as_ref()
                .map_or(Ok(Value::Unit), |expr| self_.visit_expression(expr))
        };

        match new_scope {
            true => self.scoped(HashMap::new(), callback),
            false => callback(self),
        }
    }

    fn visit_statement(&mut self, node: &AnalyzedStatement<'src>) -> StmtResult {
        match node {
            AnalyzedStatement::Let(node) => self.visit_let_stmt(node),
            AnalyzedStatement::Return(expr) => Err(InterruptKind::Return(
                expr.as_ref()
                    .map_or(Ok(Value::Unit), |expr| self.visit_expression(expr))?,
            )),
            AnalyzedStatement::Loop(node) => self.visit_loop_stmt(node),
            AnalyzedStatement::While(node) => self.visit_while_stmt(node),
            AnalyzedStatement::For(node) => self.visit_for_stmt(node),
            AnalyzedStatement::Break => Err(InterruptKind::Break),
            AnalyzedStatement::Continue => Err(InterruptKind::Continue),
            AnalyzedStatement::Expr(node) => self.visit_expression(node).map(|_| ()),
        }
    }

    fn visit_let_stmt(&mut self, node: &AnalyzedLetStmt<'src>) -> StmtResult {
        let value = self.visit_expression(&node.expr)?;
        self.scopes
            .last_mut()
            .expect("there should always be at least one scope")
            .insert(node.name, value.wrapped());
        Ok(())
    }

    fn visit_loop_stmt(&mut self, node: &AnalyzedLoopStmt<'src>) -> StmtResult {
        loop {
            match self.visit_block(&node.block, true) {
                Err(InterruptKind::Break) => break,
                Err(InterruptKind::Continue) => continue,
                res => res?,
            };
        }
        Ok(())
    }

    fn visit_while_stmt(&mut self, node: &AnalyzedWhileStmt<'src>) -> StmtResult {
        while self.visit_expression(&node.cond)?.unwrap_bool() {
            match self.visit_block(&node.block, true) {
                Err(InterruptKind::Break) => break,
                Err(InterruptKind::Continue) => continue,
                res => res?,
            };
        }
        Ok(())
    }

    fn visit_for_stmt(&mut self, node: &AnalyzedForStmt<'src>) -> StmtResult {
        // new scope just for the induction variable
        let init_val = self.visit_expression(&node.initializer)?;

        self.scoped(
            HashMap::from([(node.ident, init_val.wrapped())]),
            |self_| -> StmtResult {
                loop {
                    if !self_.visit_expression(&node.cond)?.unwrap_bool() {
                        break;
                    }

                    let res = self_.visit_block(&node.block, true);

                    self_.visit_expression(&node.update)?;

                    match res {
                        Err(InterruptKind::Break) => break,
                        Err(InterruptKind::Continue) => continue,
                        res => res?,
                    };
                }
                Ok(())
            },
        )
    }

    //////////////////////////////////

    fn visit_expression(&mut self, node: &AnalyzedExpression<'src>) -> ExprResult {
        match node {
            AnalyzedExpression::Block(block) => self.visit_block(block, true),
            AnalyzedExpression::If(node) => self.visit_if_expr(node),
            AnalyzedExpression::Int(num) => Ok(num.into()),
            AnalyzedExpression::Float(num) => Ok(num.into()),
            AnalyzedExpression::Bool(bool) => Ok(bool.into()),
            AnalyzedExpression::Char(num) => Ok(num.into()),
            AnalyzedExpression::Ident(node) => Ok(self.get_var(node.ident).borrow().clone()),
            AnalyzedExpression::Prefix(node) => self.visit_prefix_expr(node),
            AnalyzedExpression::Infix(node) => self.visit_infix_expr(node),
            AnalyzedExpression::Assign(node) => self.visit_assign_expr(node),
            AnalyzedExpression::Call(node) => self.visit_call_expr(node),
            AnalyzedExpression::Cast(node) => self.visit_cast_expr(node),
            AnalyzedExpression::Grouped(expr) => self.visit_expression(expr),
        }
    }

    fn visit_if_expr(&mut self, node: &AnalyzedIfExpr<'src>) -> ExprResult {
        if self.visit_expression(&node.cond)?.unwrap_bool() {
            self.visit_block(&node.then_block, true)
        } else if let Some(else_block) = &node.else_block {
            self.visit_block(else_block, true)
        } else {
            Ok(Value::Unit)
        }
    }

    fn visit_prefix_expr(&mut self, node: &AnalyzedPrefixExpr<'src>) -> ExprResult {
        let val = self.visit_expression(&node.expr)?;
        match node.op {
            PrefixOp::Not => Ok(!val),
            PrefixOp::Neg => Ok(-val),
            PrefixOp::Ref => match &node.expr {
                AnalyzedExpression::Ident(ident_expr) => {
                    Ok(Value::Ptr(self.get_var(ident_expr.ident)))
                }
                _ => unreachable!("the analyzer only allows referencing identifiers"),
            },
            PrefixOp::Deref => Ok(val.unwrap_ptr().borrow().clone()),
        }
    }

    fn visit_infix_expr(&mut self, node: &AnalyzedInfixExpr<'src>) -> ExprResult {
        match node.op {
            InfixOp::And => {
                return if !self.visit_expression(&node.lhs)?.unwrap_bool() {
                    Ok(false.into())
                } else {
                    self.visit_expression(&node.rhs)
                };
            }
            InfixOp::Or => {
                return if self.visit_expression(&node.lhs)?.unwrap_bool() {
                    Ok(true.into())
                } else {
                    self.visit_expression(&node.rhs)
                };
            }
            _ => {}
        }

        let lhs = self.visit_expression(&node.lhs)?;
        let rhs = self.visit_expression(&node.rhs)?;
        match node.op {
            InfixOp::Plus => Ok(lhs + rhs),
            InfixOp::Minus => Ok(lhs - rhs),
            InfixOp::Mul => Ok(lhs * rhs),
            InfixOp::Div => Ok((lhs / rhs)?),
            InfixOp::Rem => Ok((lhs % rhs)?),
            InfixOp::Pow => Ok(lhs.pow(rhs)),
            InfixOp::Eq => Ok((lhs == rhs).into()),
            InfixOp::Neq => Ok((lhs != rhs).into()),
            InfixOp::Lt => Ok((lhs < rhs).into()),
            InfixOp::Gt => Ok((lhs > rhs).into()),
            InfixOp::Lte => Ok((lhs <= rhs).into()),
            InfixOp::Gte => Ok((lhs >= rhs).into()),
            InfixOp::Shl => Ok((lhs << rhs)?),
            InfixOp::Shr => Ok((lhs >> rhs)?),
            InfixOp::BitOr => Ok(lhs | rhs),
            InfixOp::BitAnd => Ok(lhs & rhs),
            InfixOp::BitXor => Ok(lhs ^ rhs),
            InfixOp::And | InfixOp::Or => unreachable!("logical `and` and `or` are matched above"),
        }
    }

    fn visit_assign_expr(&mut self, node: &AnalyzedAssignExpr<'src>) -> ExprResult {
        let rhs = self.visit_expression(&node.expr)?;
        let mut var = self.get_var(node.assignee);
        for _ in 0..node.assignee_ptr_count {
            let new_ptr = var.borrow().clone().unwrap_ptr();
            var = new_ptr;
        }

        let new_val = match node.op {
            AssignOp::Basic => rhs,
            AssignOp::Plus => var.borrow().clone() + rhs,
            AssignOp::Minus => var.borrow().clone() - rhs,
            AssignOp::Mul => var.borrow().clone() * rhs,
            AssignOp::Div => (var.borrow().clone() / rhs)?,
            AssignOp::Rem => (var.borrow().clone() % rhs)?,
            AssignOp::Pow => var.borrow().clone().pow(rhs),
            AssignOp::Shl => (var.borrow().clone() << rhs)?,
            AssignOp::Shr => (var.borrow().clone() >> rhs)?,
            AssignOp::BitOr => var.borrow().clone() | rhs,
            AssignOp::BitAnd => var.borrow().clone() & rhs,
            AssignOp::BitXor => var.borrow().clone() ^ rhs,
        };
        *var.borrow_mut() = new_val;

        Ok(Value::Unit)
    }

    fn visit_call_expr(&mut self, node: &AnalyzedCallExpr<'src>) -> ExprResult {
        let args = node
            .args
            .iter()
            .map(|expr| self.visit_expression(expr))
            .collect::<Result<_, _>>()?;
        self.call_func(node.func, args)
    }

    fn visit_cast_expr(&mut self, node: &AnalyzedCastExpr<'src>) -> ExprResult {
        let val = self.visit_expression(&node.expr)?;
        match (val, node.type_) {
            (val @ Value::Int(_), Type::Int(0))
            | (val @ Value::Float(_), Type::Float(0))
            | (val @ Value::Char(_), Type::Char(0))
            | (val @ Value::Bool(_), Type::Bool(0)) => Ok(val),

            (Value::Int(int), Type::Float(0)) => Ok((int as f64).into()),
            (Value::Int(int), Type::Bool(0)) => Ok((int != 0).into()),
            (Value::Int(int), Type::Char(0)) => Ok((int.clamp(0, 127) as u8).into()),
            (Value::Float(float), Type::Int(0)) => Ok((float as i64).into()),
            (Value::Float(float), Type::Bool(0)) => Ok((float != 0.0).into()),
            (Value::Float(float), Type::Char(0)) => Ok((float.clamp(0.0, 127.0) as u8).into()),
            (Value::Bool(bool), Type::Int(0)) => Ok((bool as i64).into()),
            (Value::Bool(bool), Type::Float(0)) => Ok((bool as u8 as f64).into()),
            (Value::Bool(bool), Type::Char(0)) => Ok((bool as u8).into()),
            (Value::Char(char), Type::Int(0)) => Ok((char as i64).into()),
            (Value::Char(char), Type::Float(0)) => Ok((char as f64).into()),
            (Value::Char(char), Type::Bool(0)) => Ok((char != 0).into()),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

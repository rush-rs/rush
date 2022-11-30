use std::{borrow::Cow, cell::Cell, collections::HashMap, rc::Rc};

use rush_analyzer::{ast::*, AssignOp, InfixOp, PrefixOp, Type};

use crate::{InterruptKind, Value};

type ExprResult = Result<Value, InterruptKind>;
type StmtResult = Result<(), InterruptKind>;
type Scope<'src> = HashMap<&'src str, Cell<Value>>;

#[derive(Debug, Default)]
pub struct Interpreter<'src> {
    scopes: Vec<Scope<'src>>,
    functions: HashMap<&'src str, Rc<AnalyzedFunctionDefinition<'src>>>,
}

impl<'src> Interpreter<'src> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(mut self, tree: AnalyzedProgram<'src>) -> Result<i64, Cow<'static, str>> {
        for func in tree.functions.into_iter().filter(|f| f.used) {
            self.functions.insert(func.name, func.into());
        }

        let mut global_scope = HashMap::new();
        for global in tree.globals {
            global_scope.insert(
                global.name,
                match global.expr {
                    AnalyzedExpression::Int(num) => Value::Int(num).into(),
                    AnalyzedExpression::Float(num) => Value::Float(num).into(),
                    AnalyzedExpression::Bool(bool) => Value::Bool(bool).into(),
                    AnalyzedExpression::Char(num) => Value::Char(num).into(),
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

    fn get_var(&self, name: &'src str) -> &Cell<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(var) = scope.get(name) {
                return var;
            }
        }
        unreachable!("the analyzer guarantees valid variable references")
    }

    //////////////////////////////////

    fn call_func(&mut self, func_name: &'src str, args: Vec<Value>) -> ExprResult {
        if func_name == "exit" {
            return Err(InterruptKind::Exit(args[0].unwrap_int()));
        }

        let func = Rc::clone(&self.functions[func_name]);

        let mut scope = HashMap::new();
        for (param, arg) in func.params.iter().zip(args) {
            scope.insert(param.name, arg.into());
        }

        self.scopes.push(scope);
        let res = match self.visit_block(&func.block, false) {
            Ok(val) => val,
            Err(interrupt) => interrupt.into_value()?,
        };
        self.scopes.pop();
        Ok(res)
    }

    fn visit_block(&mut self, node: &AnalyzedBlock<'src>, new_scope: bool) -> ExprResult {
        if new_scope {
            self.scopes.push(HashMap::new());
        }
        for stmt in &node.stmts {
            self.visit_statement(stmt)?;
        }
        let res = node
            .expr
            .as_ref()
            .map_or(Ok(Value::Unit), |expr| self.visit_expression(expr));
        if new_scope {
            self.scopes.pop();
        }
        res
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
            .insert(node.name, value.into());
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
        let mut scope = HashMap::new();
        let init_val = self.visit_expression(&node.initializer)?;
        scope.insert(node.ident, init_val.into());

        loop {
            self.scopes.push(scope);

            if !self.visit_expression(&node.cond)?.unwrap_bool() {
                break;
            }

            let res = self.visit_block(&node.block, true);

            self.visit_expression(&node.update)?;

            scope = self
                .scopes
                .pop()
                .expect("there should always be at least one scope");

            match res {
                Err(InterruptKind::Break) => break,
                Err(InterruptKind::Continue) => continue,
                res => res?,
            };
        }
        Ok(())
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
            AnalyzedExpression::Ident(name) => Ok(self.get_var(name.ident).get()),
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
        let var = self.get_var(node.assignee);
        let new_val = match node.op {
            AssignOp::Basic => rhs,
            AssignOp::Plus => rhs + var.get(),
            AssignOp::Minus => rhs - var.get(),
            AssignOp::Mul => rhs * var.get(),
            AssignOp::Div => (rhs / var.get())?,
            AssignOp::Rem => (rhs % var.get())?,
            AssignOp::Pow => rhs.pow(var.get()),
            AssignOp::Shl => (rhs << var.get())?,
            AssignOp::Shr => (rhs >> var.get())?,
            AssignOp::BitOr => rhs | var.get(),
            AssignOp::BitAnd => rhs & var.get(),
            AssignOp::BitXor => rhs ^ var.get(),
        };
        var.set(new_val);

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
            (Value::Int(_), Type::Int)
            | (Value::Float(_), Type::Float)
            | (Value::Char(_), Type::Char)
            | (Value::Bool(_), Type::Bool) => Ok(val),

            (Value::Int(int), Type::Float) => Ok((int as f64).into()),
            (Value::Int(int), Type::Bool) => Ok((int != 0).into()),
            (Value::Int(int), Type::Char) => Ok((int.clamp(0, 127) as u8).into()),
            (Value::Float(float), Type::Int) => Ok((float as i64).into()),
            (Value::Float(float), Type::Bool) => Ok((float != 0.0).into()),
            (Value::Float(float), Type::Char) => Ok((float.clamp(0.0, 127.0) as u8).into()),
            (Value::Bool(bool), Type::Int) => Ok((bool as i64).into()),
            (Value::Bool(bool), Type::Float) => Ok((bool as u8 as f64).into()),
            (Value::Bool(bool), Type::Char) => Ok((bool as u8).into()),
            (Value::Char(char), Type::Int) => Ok((char as i64).into()),
            (Value::Char(char), Type::Float) => Ok((char as f64).into()),
            (Value::Char(char), Type::Bool) => Ok((char != 0).into()),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

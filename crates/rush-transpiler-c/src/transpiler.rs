use std::{
    collections::{HashMap, HashSet, VecDeque},
    mem,
};

use rush_analyzer::{ast::*, AssignOp, InfixOp, PrefixOp, Type};

use crate::c_ast::*;

macro_rules! comment {
    ($self:ident, $vec:expr, $msg:expr) => {
        if $self.emit_comments {
            $vec.push(Statement::Comment($msg))
        }
    };
}

pub struct Transpiler<'src> {
    /// Specifies whether the transpiler is currently inside the `main` fn.
    in_main_fn: bool,
    /// The first element is the global scope while last element is the current scope.
    scopes: Vec<HashMap<&'src str, String>>,
    /// Maps a function's name to a mangeled name.
    funcs: HashMap<&'src str, String>,
    /// Counter which is increased if a variable is declared.
    let_cnt: usize,
    /// Specifies which header files need to be included.
    required_includes: HashSet<&'static str>,
    /// If set to `true`, the transpiler will emit some comments in the `C` code.
    emit_comments: bool,
    /// The first element is the most outer loop while the last element is the current loop.
    loops: Vec<Loop>,
    /// Counter for `break` labels which is increased during loop generation.
    break_label_cnt: usize,
    /// Specifies which functions from the corelib are required.
    required_corelib_functions: HashSet<&'static str>,
}

struct Loop {
    head_label: String,
    break_label: String,
}

impl<'src> Transpiler<'src> {
    /// Creates a new [`Transpiler`].
    pub fn new(emit_comments: bool) -> Self {
        let mut required_includes = HashSet::new();
        // usages of booleans are hard to track, therefore `stdbool.h` is always included
        required_includes.insert("stdbool.h");

        Self {
            in_main_fn: false,
            scopes: vec![HashMap::new()],
            funcs: HashMap::new(),
            let_cnt: 0,
            required_includes,
            emit_comments,
            loops: vec![],
            break_label_cnt: 0,
            required_corelib_functions: HashSet::new(),
        }
    }

    /// Helper function for creating variable identifiers.
    /// Automatically generates an identifier and increases the `let_cnt`.
    /// Inserts the variable's name into the current scope and returns the ident.
    fn insert_into_scope(&mut self, name: &'src str) -> String {
        let ident = format!("{name}{}", self.let_cnt);
        self.let_cnt += 1;
        self.scopes
            .last_mut()
            .expect("there is always a scope")
            .insert(name, ident.clone());
        ident
    }

    /// Helper function for getting the mangeled name from a pure identifier.
    fn resolve_name(&'src self, name: &str) -> &'src str {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return value;
            }
        }
        unreachable!("the analyzer guarantees valid variable refrences")
    }

    pub fn transpile(&mut self, tree: AnalyzedProgram<'src>) -> CProgram {
        let globals = tree
            .globals
            .into_iter()
            .flat_map(|g| self.let_stmt(g))
            .collect();

        for func in &tree.functions {
            self.fn_signature(func)
        }

        let mut functions: VecDeque<FnDefinition> = tree
            .functions
            .into_iter()
            .filter(|f| f.used)
            .map(|f| self.fn_declaration(f))
            .collect();

        self.in_main_fn = true;
        functions.push_back(self.fn_declaration(AnalyzedFunctionDefinition {
            used: true,
            name: "main",
            params: vec![],
            return_type: Type::Int(0),
            block: tree.main_fn,
        }));
        self.in_main_fn = false;

        // TODO: prettify this code
        if self
            .required_corelib_functions
            .contains("__rush_internal_pow_int")
        {
            let (tree, _) = rush_analyzer::analyze(include_str!("./pow.rush"), "pow.rush")
                .expect("this is valid rush code");

            let func = self.fn_declaration(
                tree.functions
                    .into_iter()
                    .find(|f| f.name == "__rush_internal_pow_int")
                    .expect("this function exists"),
            );

            functions.push_front(func)
        }

        if self
            .required_corelib_functions
            .contains("__rush_internal_cast_float_to_char")
        {
            let (tree, _) = rush_analyzer::analyze(include_str!("./char.rush"), "char.rush")
                .expect("this is valid rush code");

            let func = self.fn_declaration(
                tree.functions
                    .into_iter()
                    .find(|f| f.name == "__rush_internal_cast_float_to_char")
                    .expect("this function exists"),
            );

            functions.push_front(func)
        }

        if self
            .required_corelib_functions
            .contains("__rush_internal_add_char")
        {
            let (tree, _) = rush_analyzer::analyze(include_str!("./char.rush"), "char.rush")
                .expect("this is valid rush code");

            let func = self.fn_declaration(
                tree.functions
                    .into_iter()
                    .find(|f| f.name == "__rush_internal_add_char")
                    .expect("this function exists"),
            );

            functions.push_front(func)
        }

        if self
            .required_corelib_functions
            .contains("__rush_internal_sub_char")
        {
            let (tree, _) = rush_analyzer::analyze(include_str!("./char.rush"), "char.rush")
                .expect("this is valid rush code");

            let func = self.fn_declaration(
                tree.functions
                    .into_iter()
                    .find(|f| f.name == "__rush_internal_sub_char")
                    .expect("this function exists"),
            );

            functions.push_front(func)
        }

        if self
            .required_corelib_functions
            .contains("__rush_internal_cast_int_to_char")
            || self
                .required_corelib_functions
                .contains("__rush_internal_add_char")
            || self
                .required_corelib_functions
                .contains("__rush_internal_sub_char")
        {
            let (tree, _) = rush_analyzer::analyze(include_str!("./char.rush"), "char.rush")
                .expect("this is valid rush code");

            let func = self.fn_declaration(
                tree.functions
                    .into_iter()
                    .find(|f| f.name == "__rush_internal_cast_int_to_char")
                    .expect("this function exists"),
            );

            functions.push_front(func)
        }

        CProgram {
            includes: mem::take(&mut self.required_includes),
            globals,
            functions,
        }
    }

    /// Required for adding the function prototypes first.
    /// In rush, order of functions is irrelevant.
    /// Therefore, the `C` code must also not rely on function order.
    fn fn_signature(&mut self, node: &AnalyzedFunctionDefinition<'src>) {
        let name = format!("{name}{cnt}", name = node.name, cnt = self.funcs.len());
        self.funcs.insert(node.name, name);
    }

    fn fn_declaration(&mut self, node: AnalyzedFunctionDefinition<'src>) -> FnDefinition {
        self.scopes.push(HashMap::new());

        let name = match node.name {
            "main" => "main".to_string(),
            "__rush_internal_pow_int"
            | "__rush_internal_cast_int_to_char"
            | "__rush_internal_cast_float_to_char"
            | "__rush_internal_add_char"
            | "__rush_internal_sub_char" => node.name.to_string(),
            _ => self
                .funcs
                .get(node.name)
                .expect("declared previously")
                .to_string(),
        };

        let params = node
            .params
            .into_iter()
            .filter_map(|p| {
                let ident = self.insert_into_scope(p.name);

                match p.type_ {
                    Type::Int(_) | Type::Float(_) | Type::Bool(_) | Type::Char(_) => {
                        Some((ident, p.type_.into()))
                    }
                    _ => None,
                }
            })
            .collect();

        let mut body: Vec<Statement> = node
            .block
            .stmts
            .into_iter()
            .flat_map(|s| self.statement(s))
            .collect();

        if let Some(expr) = node.block.expr {
            let (mut stmts, expr) = self.expression(expr);
            body.append(&mut stmts);
            let mut stmts = match (self.in_main_fn, expr) {
                (true, Some(expr)) => {
                    vec![
                        Statement::Expr(expr),
                        Statement::Return(Some(Expression::Int(0))),
                    ]
                }
                (true, None) => vec![Statement::Return(Some(Expression::Int(0)))],
                (false, expr) => vec![Statement::Return(expr)],
            };
            body.append(&mut stmts);
        };

        self.scopes.pop();

        FnDefinition {
            name,
            type_: node.return_type.into(),
            params,
            body,
        }
    }

    fn statement(&mut self, node: AnalyzedStatement<'src>) -> Vec<Statement> {
        match node {
            AnalyzedStatement::Let(node) => self.let_stmt(node),
            AnalyzedStatement::Return(node) => self.return_stmt(node),
            AnalyzedStatement::Loop(node) => self.loop_stmt(node),
            AnalyzedStatement::While(node) => self.while_stmt(node),
            AnalyzedStatement::For(node) => self.for_stmt(node),
            // for `break` and `continue` jumps, `goto` is used because of rush's semantics
            AnalyzedStatement::Break => {
                let loop_ = self.loops.last_mut().expect("there is always a loop");
                vec![Statement::Goto(loop_.break_label.clone())]
            }
            AnalyzedStatement::Continue => {
                let loop_ = self.loops.last_mut().expect("there is always a loop");
                vec![Statement::Goto(loop_.head_label.clone())]
            }
            AnalyzedStatement::Expr(node) => {
                let (mut stmts, expr) = self.expression(node);
                if let Some(expr) = expr {
                    stmts.push(Statement::Expr(expr));
                }
                stmts
            }
        }
    }

    fn let_stmt(&mut self, node: AnalyzedLetStmt<'src>) -> Vec<Statement> {
        let type_ = node.expr.result_type().into();
        let (mut stmts, expr) = self.expression(node.expr);

        let name = self.insert_into_scope(node.name);

        if let Some(expr) = expr {
            let stmt = match type_ {
                CType::Void => Statement::Expr(expr),
                _ => Statement::VarDeclaration(VarDeclaration { name, type_, expr }),
            };
            stmts.push(stmt)
        }

        stmts
    }

    fn return_stmt(&mut self, node: Option<AnalyzedExpression<'src>>) -> Vec<Statement> {
        match (node, self.in_main_fn) {
            (Some(expr), _) => {
                let (mut stmts, expr) = self.expression(expr);
                stmts.push(Statement::Return(expr));
                stmts
            }
            (None, false) => vec![Statement::Return(None)],
            (None, true) => vec![Statement::Return(Some(Expression::Int(0)))],
        }
    }

    fn loop_stmt(&mut self, node: AnalyzedLoopStmt<'src>) -> Vec<Statement> {
        let break_label = format!("break_{}", self.break_label_cnt);
        let head_label = format!("head_{}", self.break_label_cnt);
        self.break_label_cnt += 1;
        self.loops.push(Loop {
            head_label: head_label.clone(),
            break_label: break_label.clone(),
        });

        let mut body = vec![];
        comment!(self, body, "loop");
        body.push(Statement::Label(head_label.clone()));

        let mut stmts = match self.block_expr(node.block) {
            (mut stmts, Some(expr)) => {
                stmts.push(Statement::Expr(expr));
                stmts
            }
            (stmts, None) => stmts,
        };
        body.append(&mut stmts);
        body.push(Statement::Goto(head_label));
        body.push(Statement::Label(break_label));

        self.loops.pop();

        body
    }

    fn while_stmt(&mut self, node: AnalyzedWhileStmt<'src>) -> Vec<Statement> {
        let break_label = format!("break_{}", self.break_label_cnt);
        let head_label = format!("head_{}", self.break_label_cnt);
        self.break_label_cnt += 1;

        let mut cond_stmts = vec![];
        comment!(self, cond_stmts, "while");
        cond_stmts.push(Statement::Label(head_label.clone()));

        let (mut stmts, cond) = match self.expression(node.cond) {
            (stmts, Some(expr)) => (stmts, expr),
            (stmts, None) => return stmts,
        };
        cond_stmts.append(&mut stmts);

        let cond_check = Statement::If(IfStmt {
            cond: Expression::Prefix(Box::new(PrefixExpr {
                expr: Expression::Grouped(Box::new(cond)),
                op: PrefixOp::Not,
            })),
            then_block: vec![Statement::Goto(break_label.clone())],
            else_block: None,
        });

        cond_stmts.push(cond_check);

        self.loops.push(Loop {
            head_label: head_label.clone(),
            break_label: break_label.to_string(),
        });

        let mut body = match self.block_expr(node.block) {
            (mut stmts, Some(expr)) => {
                stmts.push(Statement::Expr(expr));
                stmts
            }
            (stmts, None) => stmts,
        };
        body.append(&mut stmts);
        cond_stmts.append(&mut body);

        self.loops.pop();

        cond_stmts.push(Statement::Goto(head_label));
        cond_stmts.push(Statement::Label(break_label));

        cond_stmts
    }

    fn for_stmt(&mut self, node: AnalyzedForStmt<'src>) -> Vec<Statement> {
        self.scopes.push(HashMap::new());
        let type_ = node.initializer.result_type().into();

        let (mut init_stmts, init) = self.expression(node.initializer);
        if let Some(expr) = init {
            let name = self.insert_into_scope(node.ident);
            let binding = Statement::VarDeclaration(VarDeclaration { name, type_, expr });
            init_stmts.push(binding)
        }

        let break_label = format!("break_{}", self.break_label_cnt);
        let head_label = format!("head_{}", self.break_label_cnt);
        let continue_label = format!("continue_{}", self.break_label_cnt);
        self.break_label_cnt += 1;

        let mut body = vec![];
        comment!(self, init_stmts, "for");
        init_stmts.push(Statement::Label(head_label.clone()));

        let (mut stmts, cond) = match self.expression(node.cond) {
            (stmts, Some(expr)) => (stmts, expr),
            (mut stmts, None) => {
                self.scopes.pop();
                init_stmts.append(&mut stmts);
                return init_stmts;
            }
        };
        body.append(&mut stmts);

        let (mut upd_stmts, upd_) = self.expression(node.update);
        if let Some(expr) = upd_ {
            upd_stmts.push(Statement::Expr(expr));
        }

        let cond_check = Statement::If(IfStmt {
            cond: Expression::Prefix(Box::new(PrefixExpr {
                expr: Expression::Grouped(Box::new(cond)),
                op: PrefixOp::Not,
            })),
            then_block: vec![Statement::Goto(break_label.clone())],
            else_block: None,
        });

        body.push(cond_check);

        self.loops.push(Loop {
            head_label: continue_label.clone(),
            break_label: break_label.clone(),
        });

        let mut stmts = match self.block_expr(node.block) {
            (mut stmts, Some(expr)) => {
                stmts.push(Statement::Expr(expr));
                stmts
            }
            (stmts, None) => stmts,
        };

        body.append(&mut stmts);
        body.push(Statement::Label(continue_label));
        body.append(&mut upd_stmts);

        self.scopes.pop();
        self.loops.pop();

        init_stmts.append(&mut body);

        init_stmts.push(Statement::Goto(head_label));
        init_stmts.push(Statement::Label(break_label));
        init_stmts
    }

    fn expression(
        &mut self,
        node: AnalyzedExpression<'src>,
    ) -> (Vec<Statement>, Option<Expression>) {
        let expr = match node {
            AnalyzedExpression::Block(node) => return self.block_expr(*node),
            AnalyzedExpression::If(node) => return self.if_expr(*node),
            AnalyzedExpression::Int(value) => Some(Expression::Int(value)),
            AnalyzedExpression::Float(value) => Some(Expression::Float(value)),
            AnalyzedExpression::Bool(value) => Some(Expression::Bool(value)),
            AnalyzedExpression::Char(value) => Some(Expression::Char(value)),
            AnalyzedExpression::Ident(ident) => match ident.result_type {
                Type::Int(_) | Type::Float(_) | Type::Bool(_) | Type::Char(_) => Some(
                    Expression::Ident(self.resolve_name(ident.ident).to_string()),
                ),
                _ => None,
            },
            AnalyzedExpression::Prefix(node) => return self.prefix_expr(*node),
            AnalyzedExpression::Infix(node) => return self.infix_expr(*node),
            AnalyzedExpression::Assign(node) => return (self.assign_expr(*node), None),
            AnalyzedExpression::Call(node) => return self.call_expr(*node),
            AnalyzedExpression::Cast(node) => return self.cast_expr(*node),
            AnalyzedExpression::Grouped(node) => return self.grouped_expr(*node),
        };
        (vec![], expr)
    }

    fn block_expr(&mut self, node: AnalyzedBlock<'src>) -> (Vec<Statement>, Option<Expression>) {
        self.scopes.push(HashMap::new());
        let mut block = vec![];

        comment!(self, block, "begin block");

        let mut stmts: Vec<Statement> = node
            .stmts
            .into_iter()
            .flat_map(|s| self.statement(s))
            .collect();

        block.append(&mut stmts);

        let expr = match node.expr {
            Some(expr) => {
                let (mut expr_stmts, expr) = self.expression(expr);
                block.append(&mut expr_stmts);
                expr
            }
            None => None,
        };

        self.scopes.pop();
        comment!(self, block, "end block");

        (block, expr)
    }

    fn if_expr(&mut self, node: AnalyzedIfExpr<'src>) -> (Vec<Statement>, Option<Expression>) {
        let mut stmts = vec![];
        let (mut cond_stmts, cond) = match self.expression(node.cond) {
            (stmts, Some(cond)) => (stmts, cond),
            (stmts, None) => return (stmts, None),
        };

        stmts.append(&mut cond_stmts);

        let res_ident = match node.result_type {
            Type::Int(_) | Type::Float(_) | Type::Bool(_) | Type::Char(_) => {
                self.let_cnt += 1;
                let ident = format!("if_res{}", self.let_cnt);
                stmts.push(Statement::VarDefinition(
                    ident.clone(),
                    node.result_type.into(),
                ));
                Some(ident)
            }
            _ => None,
        };

        let then_block = match self.block_expr(node.then_block) {
            (mut stmts, Some(expr)) => {
                stmts.push(Statement::Assign(AssignStmt {
                    assignee: res_ident.clone().expect("was declared above"),
                    assignee_ptr_count: 0,
                    op: AssignOp::Basic,
                    expr,
                }));
                stmts
            }
            (stmts, None) => stmts,
        };

        let else_block = match node.else_block {
            Some(block) => match self.block_expr(block) {
                (mut stmts, Some(expr)) => {
                    stmts.push(Statement::Assign(AssignStmt {
                        assignee: res_ident.clone().expect("was declared above"),
                        assignee_ptr_count: 0,
                        op: AssignOp::Basic,
                        expr,
                    }));
                    Some(stmts)
                }
                (stmts, None) => Some(stmts),
            },
            None => None,
        };

        stmts.push(Statement::If(IfStmt {
            cond,
            then_block,
            else_block,
        }));

        let res_expr = res_ident.map(Expression::Ident);

        (stmts, res_expr)
    }

    fn prefix_expr(
        &mut self,
        node: AnalyzedPrefixExpr<'src>,
    ) -> (Vec<Statement>, Option<Expression>) {
        let (stmts, expr) = self.expression(node.expr);
        let expr = expr.map(|expr| Expression::Prefix(Box::new(PrefixExpr { expr, op: node.op })));
        (stmts, expr)
    }

    fn infix_expr(
        &mut self,
        node: AnalyzedInfixExpr<'src>,
    ) -> (Vec<Statement>, Option<Expression>) {
        let (lhs_type, rhs_type) = (node.lhs.result_type(), node.rhs.result_type());
        let (mut lhs_stmts, lhs) = self.expression(node.lhs);
        let (mut rhs_stmts, rhs) = self.expression(node.rhs);

        match (lhs_type, rhs_type, node.op) {
            (Type::Char(0), Type::Char(0), InfixOp::Plus | InfixOp::Minus)
            | (Type::Int(0), Type::Int(0), InfixOp::Pow) => {
                lhs_stmts.append(&mut rhs_stmts);
                let func = match node.op {
                    InfixOp::Plus => "__rush_internal_add_char",
                    InfixOp::Minus => "__rush_internal_sub_char",
                    InfixOp::Pow => "__rush_internal_pow_int",
                    _ => unreachable!("these operators cannot occur here"),
                };
                self.required_corelib_functions.insert(func);
                (
                    lhs_stmts,
                    Some(Expression::Call(Box::new(CallExpr {
                        func: func.to_string(),
                        args: vec![
                            lhs.expect("exprs cannot be `None `when used here"),
                            rhs.expect("exprs cannot be None when used here"),
                        ],
                    }))),
                )
            }
            (_, _, InfixOp::Or | InfixOp::And) => {
                self.let_cnt += 1;
                let ident = format!("logical_res{}", self.let_cnt);

                rhs_stmts.push(Statement::Assign(AssignStmt {
                    assignee: ident.clone(),
                    assignee_ptr_count: 0,
                    op: AssignOp::Basic,
                    expr: match rhs {
                        Some(expr) => expr,
                        // if the rhs is [`None`], use a dummy value
                        None => Expression::Bool(false),
                    },
                }));

                let mut stmts = vec![
                    Statement::VarDefinition(ident.clone(), node.result_type.into()),
                    Statement::If(IfStmt {
                        // if the lhs is [`None`], use a dummy value
                        cond: match lhs {
                            Some(expr) => match node.op == InfixOp::Or {
                                true => expr,
                                false => Expression::Prefix(Box::new(PrefixExpr {
                                    expr: Expression::Grouped(Box::new(expr)),
                                    op: PrefixOp::Not,
                                })),
                            },
                            None => Expression::Bool(true),
                        },
                        then_block: vec![Statement::Assign(AssignStmt {
                            assignee: ident.clone(),
                            assignee_ptr_count: 0,
                            op: AssignOp::Basic,
                            // when the operator is `||` return true in this branch
                            // otherwise, return `false` in this branch
                            expr: Expression::Bool(node.op == InfixOp::Or),
                        })],
                        else_block: Some(rhs_stmts),
                    }),
                ];

                lhs_stmts.append(&mut stmts);
                (lhs_stmts, Some(Expression::Ident(ident)))
            }
            (Type::Unit | Type::Never, _, _) | (_, Type::Unit | Type::Never, _) => {
                lhs_stmts.append(&mut rhs_stmts);
                if let Some(lhs) = lhs {
                    lhs_stmts.push(Statement::Expr(lhs));
                };
                if let Some(rhs) = rhs {
                    lhs_stmts.push(Statement::Expr(rhs));
                };
                (lhs_stmts, None)
            }
            (_, _, _) => {
                lhs_stmts.append(&mut rhs_stmts);
                (
                    lhs_stmts,
                    match (lhs, rhs) {
                        (None, None) => None,
                        (Some(lhs), None) => Some(lhs),
                        (None, Some(rhs)) => Some(rhs),
                        (Some(lhs), Some(rhs)) => Some(Expression::Infix(Box::new(InfixExpr {
                            lhs,
                            rhs,
                            op: node.op,
                        }))),
                    },
                )
            }
        }
    }

    fn assign_expr(&mut self, node: AnalyzedAssignExpr<'src>) -> Vec<Statement> {
        let type_ = node.expr.result_type();
        let (mut stmts, expr) = self.expression(node.expr);
        let assignee = self.resolve_name(node.assignee).to_string();

        if let Some(expr) = expr {
            match (type_, node.op) {
                (Type::Int(_), AssignOp::Pow) => {
                    self.required_corelib_functions
                        .insert("__rush_internal_pow_int");

                    stmts.push(Statement::Assign(AssignStmt {
                        assignee: assignee.clone(),
                        assignee_ptr_count: node.assignee_ptr_count,
                        op: AssignOp::Basic,
                        expr: Expression::Call(Box::new(CallExpr {
                            func: "__rush_internal_pow_int".to_string(),
                            args: vec![
                                Expression::Deref((node.assignee_ptr_count, assignee)),
                                expr,
                            ],
                        })),
                    }))
                }
                (Type::Char(_), AssignOp::Plus | AssignOp::Minus) => {
                    let func = match node.op == AssignOp::Plus {
                        true => "__rush_internal_add_char",
                        false => "__rush_internal_sub_char",
                    };
                    self.required_corelib_functions.insert(func);

                    stmts.push(Statement::Assign(AssignStmt {
                        assignee: assignee.clone(),
                        assignee_ptr_count: node.assignee_ptr_count,
                        op: AssignOp::Basic,
                        expr: Expression::Call(Box::new(CallExpr {
                            func: func.to_string(),
                            args: vec![
                                Expression::Deref((node.assignee_ptr_count, assignee)),
                                expr,
                            ],
                        })),
                    }))
                }
                (_, op) => {
                    stmts.push(Statement::Assign(AssignStmt {
                        assignee,
                        assignee_ptr_count: node.assignee_ptr_count,
                        op,
                        expr,
                    }));
                }
            }
        }

        stmts
    }

    fn call_expr(&mut self, node: AnalyzedCallExpr<'src>) -> (Vec<Statement>, Option<Expression>) {
        let func = match node.func {
            "main" => node.func.to_string(),
            "exit" => {
                self.required_includes.insert("stdlib.h");
                node.func.to_string()
            }
            _ => self
                .funcs
                .get(node.func)
                .expect("the analyzer guarantees valid function calls")
                .to_string(),
        };

        let mut stmts = vec![];
        let mut none_arg = false;
        let args = node
            .args
            .into_iter()
            .filter_map(|expr| {
                let type_ = expr.result_type();
                let (mut expr_stmts, expr) = self.expression(expr);
                stmts.append(&mut expr_stmts);

                if expr.is_none() && type_ != Type::Unit {
                    none_arg = true;
                }

                expr
            })
            .collect();

        let expr = Box::new(CallExpr { func, args });

        match node.result_type {
            Type::Int(_) | Type::Float(_) | Type::Bool(_) | Type::Char(_) => {
                let expr = match none_arg {
                    true => None,
                    false => Some(Expression::Call(expr)),
                };
                (stmts, expr)
            }
            _ => {
                if !none_arg {
                    stmts.push(Statement::Expr(Expression::Call(expr)));
                }
                (stmts, None)
            }
        }
    }

    fn cast_expr(&mut self, node: AnalyzedCastExpr<'src>) -> (Vec<Statement>, Option<Expression>) {
        let type_ = node.expr.result_type();
        let (stmts, expr) = match self.expression(node.expr) {
            (stmts, Some(expr)) => (stmts, expr),
            (stmts, None) => return (stmts, None),
        };

        let expr = match (type_, node.type_) {
            (Type::Int(0), Type::Char(0)) | (Type::Float(0), Type::Char(0)) => {
                let func = match type_ == Type::Int(0) {
                    true => "__rush_internal_cast_int_to_char",
                    false => "__rush_internal_cast_float_to_char",
                };

                self.required_corelib_functions.insert(func);

                Expression::Call(Box::new(CallExpr {
                    func: func.to_string(),
                    args: vec![expr],
                }))
            }
            _ => Expression::Cast(Box::new(CastExpr {
                expr,
                type_: node.type_.into(),
            })),
        };

        (stmts, Some(expr))
    }

    fn grouped_expr(
        &mut self,
        node: AnalyzedExpression<'src>,
    ) -> (Vec<Statement>, Option<Expression>) {
        let (stmts, expr) = self.expression(node);
        (stmts, expr.map(|expr| Expression::Grouped(Box::new(expr))))
    }
}

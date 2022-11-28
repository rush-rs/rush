use std::{collections::HashMap, vec};

use rush_analyzer::{ast::*, AssignOp};

use crate::{
    instruction::{Instruction, Type as InstructionType},
    value::Value,
};

pub(crate) struct Compiler<'src> {
    /// The first item is the main function: execution will start here
    functions: Vec<Vec<Instruction>>,
    /// points to the current function in the functions Vec.
    fp: usize,
    // Maps a function name to its position in the `functions` Vec
    fn_names: HashMap<&'src str, usize>,

    /// Maps a name to a global index.
    globals: HashMap<&'src str, usize>,
    global_idx: usize,

    /// Contains the scopes of the current function. The last item is the current scope.
    scopes: Vec<Scope<'src>>,

    curr_fn: Function,
}

#[derive(Default)]
struct Scope<'src> {
    /// Maps an ident to a stack offset
    vars: HashMap<&'src str, usize>,
}

#[derive(Default)]
struct Function {
    /// Counter for let bindings.
    pub let_cnt: usize,
}

impl<'src> Compiler<'src> {
    pub(crate) fn new() -> Self {
        Self {
            // allocate the `prelude` and the `main` fn
            functions: vec![vec![], vec![]],
            fp: 0,
            fn_names: HashMap::new(),
            curr_fn: Function::default(),
            scopes: vec![],
            globals: HashMap::new(),
            global_idx: 0,
        }
    }

    #[inline]
    /// Emits a new instruction and appends it to the `instructions` [`Vec`].
    fn insert(&mut self, instruction: Instruction) {
        self.functions[self.fp].push(instruction)
    }

    #[inline]
    /// Returns a mutable reference to the current scope
    fn scope_mut(&mut self) -> &mut Scope<'src> {
        self.scopes.last_mut().expect("there is always a scope")
    }

    #[inline]
    /// Returns a reference to the current scope
    fn scope(&self) -> &Scope {
        self.scopes.last().expect("there is always a scope")
    }

    /// Loads the value of the specified variable name on the stack
    fn load_var(&mut self, name: &'src str) {
        let mut idx = None;
        for scope in self.scopes.iter().rev() {
            if let Some(i) = scope.vars.get(name) {
                idx = Some(i);
                break;
            };
        }

        match idx {
            Some(idx) => self.insert(Instruction::GetVar(*idx)),
            None => {
                let idx = self.globals.get(name).expect("every variable was declared");
                self.insert(Instruction::GetGlob(*idx));
            }
        }
    }

    pub(crate) fn compile(mut self, ast: &'src AnalyzedProgram) -> Vec<Vec<Instruction>> {
        // add function signatures for later use
        for (idx, func) in ast.functions.iter().filter(|f| f.used).enumerate() {
            self.fn_names.insert(func.name, idx + 2);
        }

        // add global variables
        for var in &ast.globals {
            self.declare_global(var);
        }

        // call the main fn
        self.insert(Instruction::Call(1));

        self.fp += 1;
        self.main_fn(&ast.main_fn);

        for func in ast.functions.iter().filter(|f| f.used) {
            self.functions.push(vec![]);
            self.fp += 1;
            self.fn_declaration(func);
        }

        self.functions
    }

    fn declare_global(&mut self, node: &'src AnalyzedLetStmt<'src>) {
        // map the name to the new global index
        self.globals.insert(node.name, self.global_idx);
        // push global value onto the stack
        self.expression(&node.expr);
        // pop and set the value as global
        self.insert(Instruction::SetGlob(self.global_idx));
        // increment global index
        self.global_idx += 1;
    }

    fn fn_declaration(&mut self, node: &'src AnalyzedFunctionDefinition) {
        self.curr_fn = Function::default();
        self.scopes.push(Scope::default());

        for (idx, param) in node.params.iter().enumerate().rev() {
            self.scope_mut().vars.insert(param.name, idx);
            self.insert(Instruction::SetVar(idx));
            self.curr_fn.let_cnt += 1;
        }

        self.fn_block(&node.block);
        self.scopes.pop();
        self.insert(Instruction::Ret);
    }

    fn main_fn(&mut self, node: &'src AnalyzedBlock) {
        self.curr_fn = Function::default();
        // allows main function recursion
        self.fn_names.insert("main", self.fp);
        self.block(node);
    }

    /// Similar to `self.block` but does not push a new [`Scope`].
    fn fn_block(&mut self, node: &'src AnalyzedBlock) {
        for stmt in &node.stmts {
            self.statement(stmt);
        }
        match &node.expr {
            Some(expr) => self.expression(expr),
            None => {
                self.insert(Instruction::Push(Value::Unit));
            }
        }
    }

    /// Compiles a block of statements.
    /// Results in the optional expr (unit if there is none).
    /// Automatically pushes a new [`Scope`] for the block.
    fn block(&mut self, node: &'src AnalyzedBlock) {
        self.scopes.push(Scope::default());
        for stmt in &node.stmts {
            self.statement(stmt);
        }
        match &node.expr {
            Some(expr) => self.expression(expr),
            None => {
                self.insert(Instruction::Push(Value::Unit));
            }
        }
        self.scopes.pop();
    }

    fn statement(&mut self, node: &'src AnalyzedStatement) {
        match node {
            AnalyzedStatement::Let(node) => self.let_stmt(node),
            AnalyzedStatement::Return(expr) => {
                if let Some(expr) = expr {
                    self.expression(expr);
                }
                self.insert(Instruction::Ret);
            }
            AnalyzedStatement::Loop(_) => todo!(),
            AnalyzedStatement::While(_) => todo!(),
            AnalyzedStatement::For(_) => todo!(),
            AnalyzedStatement::Break => todo!(),
            AnalyzedStatement::Continue => todo!(),
            AnalyzedStatement::Expr(node) => {
                self.expression(node);
                self.insert(Instruction::Pop)
            }
        }
    }

    fn let_stmt(&mut self, node: &'src AnalyzedLetStmt) {
        self.expression(&node.expr);
        let let_cnt = self.curr_fn.let_cnt;
        self.insert(Instruction::SetVar(let_cnt));
        self.scope_mut().vars.insert(node.name, let_cnt);
        self.curr_fn.let_cnt += 1;
    }

    fn expression(&mut self, node: &'src AnalyzedExpression) {
        match node {
            AnalyzedExpression::Int(value) => self.insert(Instruction::Push(Value::Int(*value))),
            AnalyzedExpression::Float(value) => {
                self.insert(Instruction::Push(Value::Float(*value)))
            }
            AnalyzedExpression::Bool(value) => self.insert(Instruction::Push(Value::Bool(*value))),
            AnalyzedExpression::Char(value) => self.insert(Instruction::Push(Value::Char(*value))),
            AnalyzedExpression::Ident(node) => self.load_var(node.ident),
            AnalyzedExpression::Block(node) => self.block(node),
            AnalyzedExpression::If(node) => self.if_expr(node),
            AnalyzedExpression::Prefix(node) => self.prefix_expr(node),
            AnalyzedExpression::Infix(node) => self.infix_expr(node),
            AnalyzedExpression::Assign(node) => self.assign_expr(node),
            AnalyzedExpression::Call(node) => self.call_expr(node),
            AnalyzedExpression::Cast(node) => self.cast_expr(node),
            AnalyzedExpression::Grouped(node) => self.expression(node),
        }
    }

    fn if_expr(&mut self, node: &'src AnalyzedIfExpr) {
        // compile the condition
        self.expression(&node.cond);
        let after_condition = self.functions[self.fp].len() - 1;

        // compile the `then` branch
        self.block(&node.then_block);
        let after_then_idx = self.functions[self.fp].len() - 1;

        // insert the jump (skip the then block)
        self.functions[self.fp].insert(
            after_condition + 1,
            Instruction::JmpCond(after_then_idx + 3),
        );

        if let Some(else_block) = &node.else_block {
            self.block(else_block);
            let after_else = self.functions[self.fp].len();
            self.functions[self.fp].insert(after_then_idx + 2, Instruction::Jmp(after_else + 1))
        }
    }

    fn prefix_expr(&mut self, node: &'src AnalyzedPrefixExpr) {
        self.expression(&node.expr);
        self.insert(Instruction::from(node.op));
    }

    fn infix_expr(&mut self, node: &'src AnalyzedInfixExpr) {
        self.expression(&node.lhs);
        self.expression(&node.rhs);
        self.insert(Instruction::from(node.op));
    }

    fn assign_expr(&mut self, node: &'src AnalyzedAssignExpr) {
        if node.op != AssignOp::Basic {
            // load the assignee value
            self.load_var(node.assignee);
            self.expression(&node.expr);
            self.insert(Instruction::from(node.op))
        } else {
            self.expression(&node.expr);
        }

        match self.scope().vars.get(node.assignee) {
            Some(idx) => self.insert(Instruction::SetVar(*idx)),
            None => {
                let idx = self
                    .globals
                    .get(node.assignee)
                    .expect("every variable was declared");
                self.insert(Instruction::SetGlob(*idx));
            }
        };
        // result value of assignments is `()`
        self.insert(Instruction::Push(Value::Unit))
    }

    fn call_expr(&mut self, node: &'src AnalyzedCallExpr) {
        for arg in node.args.iter() {
            self.expression(arg);
        }

        match node.func {
            "exit" => self.insert(Instruction::Exit),
            func => {
                let fn_idx = self.fn_names.get(func).expect("every function exists");
                self.insert(Instruction::Call(*fn_idx));
            }
        }
    }

    fn cast_expr(&mut self, node: &'src AnalyzedCastExpr) {
        self.expression(&node.expr);
        match (node.expr.result_type(), node.type_) {
            (from, to) if from == to => {}
            (_, to) => self.insert(Instruction::Cast(InstructionType::from(to))),
        }
    }
}

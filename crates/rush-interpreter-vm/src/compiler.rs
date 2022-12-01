use std::{collections::HashMap, vec};

use rush_analyzer::{ast::*, AssignOp, InfixOp, Type};

use crate::{
    instruction::{Instruction, Program, Type as InstructionType},
    value::Value,
};

pub(crate) struct Compiler<'src> {
    /// The first item is the main function: execution will start here
    functions: Vec<Vec<Instruction>>,
    /// points to the current function in the functions Vec.
    fp: usize,
    // Maps a function name to its position in the `functions` Vec
    fn_names: HashMap<&'src str, usize>,

    /// Maps a name to a global index and the variable type.
    globals: HashMap<&'src str, (usize, Type)>,
    global_idx: usize,

    /// Contains the scopes of the current function. The last item is the current scope.
    scopes: Vec<Scope<'src>>,

    curr_fn: Function,

    /// Contains information about the current loop
    curr_loop: Loop,
}

#[derive(Default, Debug)]
struct Scope<'src> {
    /// Maps an ident to a variable
    vars: HashMap<&'src str, Variable>,
}

#[derive(Debug, Clone, Copy)]
enum Variable {
    Local { stack_idx: usize, type_: Type },
    Global,
}

#[derive(Default)]
struct Function {
    /// Counter for let bindings.
    pub let_cnt: usize,
}

#[derive(Default)]
struct Loop {
    /// Specifies the instruction index of the loop head
    head: usize,
    /// Specifies the instruction indices in the current function of `break` statements.
    /// Used for replacing the offset with the real value after the loop body has been compiled.
    break_jmp_indices: Vec<usize>,
}

impl Loop {
    fn new(head: usize) -> Self {
        Self {
            head,
            break_jmp_indices: vec![],
        }
    }
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
            curr_loop: Loop::default(),
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

    /// Returns the specified variable given its identifier
    fn resolve_var(&self, name: &'src str) -> Variable {
        for scope in self.scopes.iter().rev() {
            if let Some(i) = scope.vars.get(name) {
                return *i;
            };
        }
        Variable::Global
    }

    /// Loads the value of the specified variable name on the stack
    fn load_var(&mut self, name: &'src str) {
        let var = self.resolve_var(name);
        match var {
            Variable::Local {
                type_: Type::Unit | Type::Never,
                ..
            } => {} // ignore unit / never values
            Variable::Local { stack_idx, type_ } => self.insert(Instruction::GetVar(stack_idx)),
            Variable::Global => {
                let var = self
                    .globals
                    .get(name)
                    .expect(&format!("every variable was declared: {name}"));
                match var {
                    (_, Type::Unit | Type::Never) => {} // ignore unit / never values
                    (idx, _) => self.insert(Instruction::GetGlob(*idx)),
                }
            }
        }
    }

    pub(crate) fn compile(mut self, ast: &'src AnalyzedProgram) -> Program {
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

        Program(self.functions)
    }

    fn declare_global(&mut self, node: &'src AnalyzedLetStmt<'src>) {
        // map the name to the new global index
        self.globals
            .insert(node.name, (self.global_idx, node.expr.result_type()));
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

        for param in node.params.iter().rev() {
            let stack_idx = self.curr_fn.let_cnt;
            self.scope_mut().vars.insert(
                param.name,
                Variable::Local {
                    stack_idx,
                    type_: param.type_,
                },
            );

            if !matches!(param.type_, Type::Unit | Type::Never) {
                self.insert(Instruction::SetVar(self.curr_fn.let_cnt));
                self.curr_fn.let_cnt += 1;
            }
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
            None => {}
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
            None => {}
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
            AnalyzedStatement::Loop(node) => self.loop_stmt(node),
            AnalyzedStatement::While(node) => self.while_stmt(node),
            AnalyzedStatement::For(node) => self.for_stmt(node),
            AnalyzedStatement::Break => {
                // the jmp instruction is corrected later
                self.curr_loop
                    .break_jmp_indices
                    .push(self.functions[self.fp].len() + 1);
                self.insert(Instruction::Jmp(usize::MAX));
            }
            AnalyzedStatement::Continue => self.insert(Instruction::Jmp(self.curr_loop.head)),
            AnalyzedStatement::Expr(node) => {
                self.expression(node);
                if !matches!(node.result_type(), Type::Unit | Type::Never) {
                    self.insert(Instruction::Pop)
                }
            }
        }
    }

    fn let_stmt(&mut self, node: &'src AnalyzedLetStmt) {
        self.expression(&node.expr);

        match node.expr.result_type() {
            Type::Unit | Type::Never => {
                self.scope_mut().vars.insert(
                    node.name,
                    Variable::Local {
                        stack_idx: 0,
                        type_: Type::Unit,
                    },
                );
            }
            _ => {
                let stack_idx = self.curr_fn.let_cnt;
                self.insert(Instruction::SetVar(stack_idx));

                self.scope_mut().vars.insert(
                    node.name,
                    Variable::Local {
                        stack_idx,
                        type_: node.expr.result_type(),
                    },
                );
                self.curr_fn.let_cnt += 1;
            }
        }
    }

    /// Fills in any blank-value `break` statement instructions.
    fn fill_blank_jmps(&mut self, offset: usize) {
        for idx in &self.curr_loop.break_jmp_indices {
            match &mut self.functions[self.fp][*idx] {
                Instruction::Jmp(o) => *o = offset,
                Instruction::JmpFalse(o) => *o = offset,
                _ => unreachable!("other instructions do not jump"),
            }
        }
    }

    fn loop_stmt(&mut self, node: &'src AnalyzedLoopStmt) {
        // save location of the loop head (for continue stmts)
        let loop_head_pos = self.functions[self.fp].len();
        self.curr_loop = Loop::new(loop_head_pos);

        // compile the loop body
        self.block(&node.block);
        if node.block.expr.as_ref().map_or(false, |e| {
            !matches!(e.result_type(), Type::Unit | Type::Never)
        }) {
            self.insert(Instruction::Pop);
        }

        // jump back to the top
        self.insert(Instruction::Jmp(loop_head_pos));

        // correct placeholder break values
        self.fill_blank_jmps(self.functions[self.fp].len());
    }

    fn while_stmt(&mut self, node: &'src AnalyzedWhileStmt) {
        // save location of the loop head (for continue stmts)
        let loop_head_pos = self.functions[self.fp].len();
        self.curr_loop = Loop::new(loop_head_pos);

        // compile the while condition
        self.expression(&node.cond);

        // jump to the end if the condition is false
        self.curr_loop
            .break_jmp_indices
            .push(self.functions[self.fp].len());
        self.insert(Instruction::JmpFalse(usize::MAX));

        // compile the loop body
        self.block(&node.block);
        if node.block.expr.as_ref().map_or(false, |e| {
            !matches!(e.result_type(), Type::Unit | Type::Never)
        }) {
            self.insert(Instruction::Pop);
        }

        // jump back to the top
        self.insert(Instruction::Jmp(loop_head_pos));

        // correct placeholder break values
        self.fill_blank_jmps(self.functions[self.fp].len());
    }

    fn for_stmt(&mut self, node: &'src AnalyzedForStmt) {
        // compile the init expression
        self.expression(&node.initializer);
        let stack_idx = self.curr_fn.let_cnt;
        self.insert(Instruction::SetVar(stack_idx));
        self.scope_mut().vars.insert(
            node.ident,
            Variable::Local {
                stack_idx,
                type_: node.initializer.result_type(),
            },
        );
        self.curr_fn.let_cnt += 1;

        // save location of the loop head (for continue stmts)
        let loop_head_pos = self.functions[self.fp].len();
        self.curr_loop = Loop::new(loop_head_pos);

        // compile the condition expr
        self.expression(&node.cond);

        // jump to the end if the condition is false
        self.curr_loop
            .break_jmp_indices
            .push(self.functions[self.fp].len());
        self.insert(Instruction::JmpFalse(usize::MAX));

        // compile the loop body
        for stmt in &node.block.stmts {
            self.statement(stmt);
        }
        match &node.block.expr {
            Some(expr) => {
                self.expression(expr);
                if !matches!(expr.result_type(), Type::Unit | Type::Never) {
                    self.insert(Instruction::Pop)
                }
            }
            None => {}
        }

        // compile the update expression
        self.expression(&node.update);
        if node.block.expr.as_ref().map_or(false, |e| {
            !matches!(e.result_type(), Type::Unit | Type::Never)
        }) {
            self.insert(Instruction::Pop);
        }

        // jump back to the top
        self.insert(Instruction::Jmp(loop_head_pos));

        // correct placeholder break values
        self.fill_blank_jmps(self.functions[self.fp].len());
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
            Instruction::JmpFalse(after_then_idx + 3),
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
        match node.op {
            InfixOp::Or => {
                self.expression(&node.lhs);
                let merge_jmp_idx = self.functions[self.fp].len();
                self.insert(Instruction::JmpFalse(usize::MAX));
                self.expression(&node.rhs);
                self.functions[self.fp][merge_jmp_idx] =
                    Instruction::JmpFalse(self.functions[self.fp].len() - 1);
            }
            InfixOp::And => {
                self.expression(&node.lhs);
                self.insert(Instruction::Not);
                let merge_jmp_idx = self.functions[self.fp].len();
                self.insert(Instruction::JmpFalse(usize::MAX));
                self.expression(&node.rhs);
                self.functions[self.fp][merge_jmp_idx] =
                    Instruction::JmpFalse(self.functions[self.fp].len() - 1);
            }
            op => {
                self.expression(&node.lhs);
                self.expression(&node.rhs);
                self.insert(Instruction::from(op));
            }
        }
    }

    fn assign_expr(&mut self, node: &'src AnalyzedAssignExpr) {
        if node.op != AssignOp::Basic {
            // load the assignee value
            self.expression(&node.expr);
            self.load_var(node.assignee);
            self.insert(Instruction::from(node.op))
        } else {
            self.expression(&node.expr);
        }

        let assignee = self.resolve_var(node.assignee);
        match assignee {
            Variable::Local {
                type_: Type::Unit | Type::Never,
                ..
            } => {}
            Variable::Local { stack_idx, .. } => self.insert(Instruction::SetVar(stack_idx)),
            Variable::Global => {
                let var = self
                    .globals
                    .get(node.assignee)
                    .expect("every variable was declared");

                match var {
                    (_, Type::Unit | Type::Never) => {}
                    (idx, _) => self.insert(Instruction::SetGlob(*idx)),
                };
            }
        };
    }

    fn call_expr(&mut self, node: &'src AnalyzedCallExpr) {
        for arg in node
            .args
            .iter()
            .filter(|a| !matches!(a.result_type(), Type::Unit | Type::Never))
        {
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

use std::{collections::HashMap, vec};

use rush_analyzer::{ast::*, InfixOp, Type};

use crate::{
    instruction::{Instruction, Program, Type as InstructionType},
    value::Value,
};

#[derive(Default)]
pub struct Compiler<'src> {
    /// The first item is the prelude: execution will start here
    functions: Vec<Vec<Instruction>>,
    /// Maps a function name to its position in the `functions` Vec
    fn_names: HashMap<&'src str, usize>,

    /// Maps a name to a global index and the variable type.
    globals: HashMap<&'src str, usize>,

    /// Contains the scopes of the current function. The last item is the current scope.
    scopes: Vec<Scope<'src>>,

    /// Counter for let bindings in the current function.
    local_let_count: usize,

    /// Contains information about the current loop(s)
    loops: Vec<Loop>,
}

/// Maps idents to variables
type Scope<'src> = HashMap<&'src str, Variable>;

#[derive(Debug, Clone, Copy)]
enum Variable {
    Local { idx: usize, type_: Type },
    Global { idx: usize },
}

#[derive(Default)]
struct Loop {
    /// Specifies the instruction indices in the current function of `break` statements.
    /// Used for replacing the offset with the real value after the loop body has been compiled.
    break_jmp_indices: Vec<usize>,
    /// Specifies the instruction indices in the current function of `continue` statements.
    /// Used for replacing the offset with the real value after the loop body has been compiled.
    continue_jmp_indices: Vec<usize>,
}

impl<'src> Compiler<'src> {
    pub(crate) fn new() -> Self {
        Self {
            // begin with empty `prelude`
            functions: vec![vec![]],
            ..Default::default()
        }
    }

    #[inline]
    /// Emits a new instruction and appends it to the `instructions` [`Vec`].
    fn insert(&mut self, instruction: Instruction) {
        self.functions
            .last_mut()
            .expect("there is always a function")
            .push(instruction)
    }

    #[inline]
    /// Returns a mutable reference to the current function
    fn curr_fn_mut(&mut self) -> &mut Vec<Instruction> {
        self.functions
            .last_mut()
            .expect("there is always a function")
    }

    #[inline]
    /// Returns a mutable reference to the current scope
    fn scope_mut(&mut self) -> &mut Scope<'src> {
        self.scopes.last_mut().expect("there is always a scope")
    }

    #[inline]
    /// Returns a mutable reference to the current loop
    fn curr_loop_mut(&mut self) -> &mut Loop {
        self.loops
            .last_mut()
            .expect("there is always a loop when called")
    }

    /// Returns the specified variable given its identifier
    fn resolve_var(&self, name: &'src str) -> Variable {
        for scope in self.scopes.iter().rev() {
            if let Some(i) = scope.get(name) {
                return *i;
            };
        }
        Variable::Global {
            idx: self.globals[name],
        }
    }

    /// Loads the value of the specified variable name on the stack
    fn load_var(&mut self, name: &'src str) {
        let var = self.resolve_var(name);
        match var {
            Variable::Local {
                type_: Type::Unit | Type::Never,
                ..
            } => {} // ignore unit / never values
            Variable::Local { idx, .. } => self.insert(Instruction::GetVar(idx)),
            Variable::Global { idx } => self.insert(Instruction::GetGlobal(idx)),
        }
    }

    pub(crate) fn compile(mut self, ast: AnalyzedProgram<'src>) -> Program {
        // map function names to indices
        for (idx, func) in ast.functions.iter().filter(|f| f.used).enumerate() {
            self.fn_names.insert(func.name, idx + 2);
        }

        // add global variables
        for var in ast.globals.into_iter().filter(|g| g.used) {
            self.declare_global(var);
        }

        // call the main fn
        self.insert(Instruction::Call(1));

        // compile the main function
        self.main_fn(ast.main_fn);

        // compile all other functions
        for func in ast.functions.into_iter().filter(|f| f.used) {
            self.functions.push(vec![]);
            self.fn_declaration(func);
        }

        Program(self.functions)
    }

    fn declare_global(&mut self, node: AnalyzedLetStmt<'src>) {
        // map the name to the new global index
        let idx = self.globals.len();
        self.globals.insert(node.name, idx);
        // push global value onto the stack
        self.expression(node.expr);
        // pop and set the value as global
        self.insert(Instruction::SetGlobal(idx));
    }

    fn fn_declaration(&mut self, node: AnalyzedFunctionDefinition<'src>) {
        self.local_let_count = 0;
        self.scopes.push(Scope::default());

        for param in node.params.iter().rev() {
            let idx = self.local_let_count;
            self.scope_mut().insert(
                param.name,
                Variable::Local {
                    idx,
                    type_: param.type_,
                },
            );

            if !matches!(param.type_, Type::Unit | Type::Never) {
                self.insert(Instruction::SetVar(self.local_let_count));
                self.local_let_count += 1;
            }
        }

        self.block(node.block, false);
        self.scopes.pop();
        self.insert(Instruction::Ret);
    }

    fn main_fn(&mut self, node: AnalyzedBlock<'src>) {
        self.functions.push(vec![]);
        self.local_let_count = 0;
        self.fn_names.insert("main", 1);
        self.block(node, true);
    }

    /// Compiles a block of statements.
    /// Results in the optional expr (unit if there is none).
    /// Automatically pushes a new [`Scope`] for the block when `new_scope` is `true`.
    fn block(&mut self, node: AnalyzedBlock<'src>, new_scope: bool) {
        if new_scope {
            self.scopes.push(Scope::default());
        }
        for stmt in node.stmts {
            self.statement(stmt);
        }
        if let Some(expr) = node.expr {
            self.expression(expr);
        }
        if new_scope {
            self.scopes.pop();
        }
    }

    fn statement(&mut self, node: AnalyzedStatement<'src>) {
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
                let end = self.curr_fn_mut().len();
                self.curr_loop_mut().break_jmp_indices.push(end);
                self.insert(Instruction::Jmp(usize::MAX));
            }
            AnalyzedStatement::Continue => {
                // the jmp instruction is corrected later
                let end = self.curr_fn_mut().len();
                self.curr_loop_mut().continue_jmp_indices.push(end);
                self.insert(Instruction::Jmp(usize::MAX));
            }
            AnalyzedStatement::Expr(node) => {
                let expr_type = node.result_type();
                self.expression(node);
                if !matches!(expr_type, Type::Unit | Type::Never) {
                    self.insert(Instruction::Drop)
                }
            }
        }
    }

    fn let_stmt(&mut self, node: AnalyzedLetStmt<'src>) {
        let expr_type = node.expr.result_type();
        self.expression(node.expr);

        match expr_type {
            Type::Unit | Type::Never => {
                self.scope_mut().insert(
                    node.name,
                    // TODO: Variable::Unit or use options
                    Variable::Local {
                        idx: 0,
                        type_: Type::Unit,
                    },
                );
            }
            _ => {
                let idx = self.local_let_count;
                self.insert(Instruction::SetVar(idx));

                self.scope_mut().insert(
                    node.name,
                    Variable::Local {
                        idx,
                        type_: expr_type,
                    },
                );
                self.local_let_count += 1;
            }
        }
    }

    /// Fills in any blank-value `jmp` / `jmpfalse` instructions to point to the specified target.
    fn fill_blank_jmps(&mut self, jmps: Vec<usize>, target: usize) {
        for idx in &jmps {
            match &mut self.curr_fn_mut()[*idx] {
                Instruction::Jmp(o) => *o = target,
                Instruction::JmpFalse(o) => *o = target,
                _ => unreachable!("other instructions do not jump"),
            }
        }
    }

    fn loop_stmt(&mut self, node: AnalyzedLoopStmt<'src>) {
        // save location of the loop head (for continue stmts)
        let loop_head_pos = self.curr_fn_mut().len();
        self.loops.push(Loop::default());

        // compile the loop body
        let block_expr_type = node
            .block
            .expr
            .as_ref()
            .map_or(Type::Unit, |expr| expr.result_type());
        self.block(node.block, true);
        if !matches!(block_expr_type, Type::Unit | Type::Never) {
            self.insert(Instruction::Drop);
        }

        // jump back to the top
        self.insert(Instruction::Jmp(loop_head_pos));

        // correct placeholder `break` / `continue` values
        let loop_ = self.loops.pop().expect("pushed above"); // TODO: use helper function for
        let pos = self.curr_fn_mut().len();
        self.fill_blank_jmps(loop_.break_jmp_indices, pos);
        self.fill_blank_jmps(loop_.continue_jmp_indices, loop_head_pos);
    }

    fn while_stmt(&mut self, node: AnalyzedWhileStmt<'src>) {
        // save location of the loop head (for continue stmts)
        let loop_head_pos = self.curr_fn_mut().len();

        // compile the while condition
        self.expression(node.cond);

        // push the loop here (`continue` / `break` can be in cond)
        self.loops.push(Loop::default());

        // jump to the end if the condition is false
        let end = self.curr_fn_mut().len();
        self.curr_loop_mut().break_jmp_indices.push(end);
        self.insert(Instruction::JmpFalse(usize::MAX));

        // compile the loop body
        let block_expr_type = node
            .block
            .expr
            .as_ref()
            .map_or(Type::Unit, |expr| expr.result_type());
        self.block(node.block, true);
        if !matches!(block_expr_type, Type::Unit | Type::Never) {
            self.insert(Instruction::Drop);
        }

        // jump back to the top
        self.insert(Instruction::Jmp(loop_head_pos));

        // correct placeholder `break` / `continue` values
        let loop_ = self.loops.pop().expect("pushed above"); // TODO: helper func
        let pos = self.curr_fn_mut().len();
        self.fill_blank_jmps(loop_.break_jmp_indices, pos);
        self.fill_blank_jmps(loop_.continue_jmp_indices, loop_head_pos);
    }

    fn for_stmt(&mut self, node: AnalyzedForStmt<'src>) {
        // compile the init expression
        let initializer_type = node.initializer.result_type();
        self.expression(node.initializer);
        let idx = self.local_let_count;
        self.insert(Instruction::SetVar(idx));
        self.scope_mut().insert(
            node.ident,
            Variable::Local {
                idx,
                type_: initializer_type,
            },
        );
        self.local_let_count += 1;

        // save location of the loop head (for repetition)
        let loop_head_pos = self.curr_fn_mut().len();

        // compile the condition expr
        self.expression(node.cond);

        self.loops.push(Loop::default());

        // jump to the end of the loop if the condition is false
        let curr_pos = self.curr_fn_mut().len();
        self.curr_loop_mut().break_jmp_indices.push(curr_pos);
        self.insert(Instruction::JmpFalse(usize::MAX));

        let block_expr_type = node
            .block
            .expr
            .as_ref()
            .map_or(Type::Unit, |expr| expr.result_type());
        self.block(node.block, true);
        if !matches!(block_expr_type, Type::Unit | Type::Never) {
            self.insert(Instruction::Drop);
        }

        // correct placeholder `continue` values
        let curr_pos = self.curr_fn_mut().len();
        let loop_ = self.loops.pop().expect("pushed above");
        self.fill_blank_jmps(loop_.continue_jmp_indices, curr_pos);

        // compile the update expression
        let update_type = node.update.result_type();
        self.expression(node.update);
        if !matches!(update_type, Type::Unit | Type::Never) {
            self.insert(Instruction::Drop);
        }

        // jump back to the top
        self.insert(Instruction::Jmp(loop_head_pos));

        // correct placeholder break values
        let pos = self.curr_fn_mut().len();
        self.fill_blank_jmps(loop_.break_jmp_indices, pos);
    }

    fn expression(&mut self, node: AnalyzedExpression<'src>) {
        match node {
            AnalyzedExpression::Int(value) => self.insert(Instruction::Push(Value::Int(value))),
            AnalyzedExpression::Float(value) => self.insert(Instruction::Push(Value::Float(value))),
            AnalyzedExpression::Bool(value) => self.insert(Instruction::Push(Value::Bool(value))),
            AnalyzedExpression::Char(value) => self.insert(Instruction::Push(Value::Char(value))),
            AnalyzedExpression::Ident(node) => self.load_var(node.ident),
            AnalyzedExpression::Block(node) => self.block(*node, true),
            AnalyzedExpression::If(node) => self.if_expr(*node),
            AnalyzedExpression::Prefix(node) => self.prefix_expr(*node),
            AnalyzedExpression::Infix(node) => self.infix_expr(*node),
            AnalyzedExpression::Assign(node) => self.assign_expr(*node),
            AnalyzedExpression::Call(node) => self.call_expr(*node),
            AnalyzedExpression::Cast(node) => self.cast_expr(*node),
            AnalyzedExpression::Grouped(node) => self.expression(*node),
        }
    }

    fn if_expr(&mut self, node: AnalyzedIfExpr<'src>) {
        // compile the condition
        self.expression(node.cond);
        let after_condition = self.curr_fn_mut().len();
        self.insert(Instruction::Jmp(usize::MAX)); // placeholder

        // compile the `then` branch
        self.block(node.then_block, true);
        let after_then_idx = self.curr_fn_mut().len();

        if let Some(else_block) = node.else_block {
            self.insert(Instruction::Jmp(usize::MAX)); // placeholder
                                                       // if there is `else`, jump to the instruction after the jump after `then`
            self.curr_fn_mut()[after_condition] = Instruction::JmpFalse(after_then_idx + 1);

            self.block(else_block, true);
            let after_else = self.curr_fn_mut().len();

            // skip the `else` block when coming from the `then` block
            self.curr_fn_mut()[after_then_idx] = Instruction::Jmp(after_else);
        } else {
            // if there is no `else` branch, jump after the last instruction of the `then` branch
            self.curr_fn_mut()[after_condition] = Instruction::JmpFalse(after_then_idx);
        }
    }

    fn prefix_expr(&mut self, node: AnalyzedPrefixExpr<'src>) {
        self.expression(node.expr);
        self.insert(Instruction::from(node.op));
    }

    fn infix_expr(&mut self, node: AnalyzedInfixExpr<'src>) {
        match node.op {
            InfixOp::Or => {
                self.expression(node.lhs);
                self.insert(Instruction::Not);
                let merge_jmp_idx = self.curr_fn_mut().len();
                self.insert(Instruction::JmpFalse(usize::MAX));
                self.expression(node.rhs);
                let pos = self.curr_fn_mut().len() + 2;
                self.insert(Instruction::Jmp(pos));
                self.insert(Instruction::Push(Value::Bool(true)));
                self.curr_fn_mut()[merge_jmp_idx] =
                    Instruction::JmpFalse(self.curr_fn_mut().len() - 1);
            }
            InfixOp::And => {
                self.expression(node.lhs);
                let merge_jmp_idx = self.curr_fn_mut().len();
                self.insert(Instruction::JmpFalse(usize::MAX));
                self.expression(node.rhs);
                let pos = self.curr_fn_mut().len() + 2;
                self.insert(Instruction::Jmp(pos));
                self.insert(Instruction::Push(Value::Bool(false)));
                self.curr_fn_mut()[merge_jmp_idx] =
                    Instruction::JmpFalse(self.curr_fn_mut().len() - 1);
            }
            op => {
                self.expression(node.lhs);
                self.expression(node.rhs);
                self.insert(Instruction::from(op));
            }
        }
    }

    fn assign_expr(&mut self, node: AnalyzedAssignExpr<'src>) {
        match node.op.try_into() {
            Ok(instruction) => {
                // load the assignee value
                self.load_var(node.assignee);
                self.expression(node.expr);
                self.insert(instruction);
            }
            Err(()) => self.expression(node.expr),
        }

        let assignee = self.resolve_var(node.assignee);
        match assignee {
            Variable::Local {
                type_: Type::Unit | Type::Never,
                ..
            } => {}
            Variable::Local { idx, .. } => self.insert(Instruction::SetVar(idx)),
            Variable::Global { idx } => self.insert(Instruction::SetGlobal(idx)),
        };
    }

    fn call_expr(&mut self, node: AnalyzedCallExpr<'src>) {
        for arg in node
            .args
            .into_iter()
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

    fn cast_expr(&mut self, node: AnalyzedCastExpr<'src>) {
        let expr_type = node.expr.result_type();
        self.expression(node.expr);
        match (expr_type, node.type_) {
            (from, to) if from == to => {}
            (_, to) => self.insert(Instruction::Cast(InstructionType::from(to))),
        }
    }
}

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    vec,
};

use rush_parser::{ast::*, Span};

use crate::{ast::*, Diagnostic, DiagnosticLevel, ErrorKind};

#[derive(Default, Debug)]
pub struct Analyzer<'src> {
    functions: HashMap<&'src str, Function<'src>>,
    diagnostics: Vec<Diagnostic<'src>>,
    scopes: Vec<HashMap<&'src str, Variable<'src>>>,
    curr_func_name: &'src str,
    /// Specifies the depth of loops, `break` / `continue` legal if > 0.
    loop_count: usize,
    builtin_functions: HashMap<&'static str, BuiltinFunction>,
    /// The names of all used builtin functions
    used_builtins: HashSet<&'src str>,
    /// Specifies whether there is at least one `break` statement inside the current loop.
    current_loop_is_terminated: bool,
    /// The source code of the program to be analyzed
    source: &'src str,
}

#[derive(Debug)]
struct Function<'src> {
    pub ident: Spanned<'src, &'src str>,
    pub params: Spanned<'src, Vec<Parameter<'src>>>,
    pub return_type: Spanned<'src, Option<Type>>,
    pub used: bool,
}

#[derive(Debug, Clone)]
struct BuiltinFunction {
    param_types: Vec<Type>,
    return_type: Type,
}

impl BuiltinFunction {
    fn new(param_types: Vec<Type>, return_type: Type) -> Self {
        Self {
            param_types,
            return_type,
        }
    }
}

#[derive(Debug)]
struct Variable<'src> {
    pub type_: Type,
    pub span: Span<'src>,
    pub used: bool,
    pub mutable: bool,
    pub mutated: bool,
}

impl<'src> Analyzer<'src> {
    /// Creates a new [`Analyzer`].
    pub fn new(source: &'src str) -> Self {
        Self {
            builtin_functions: HashMap::from([(
                "exit",
                BuiltinFunction::new(vec![Type::Int(0)], Type::Never),
            )]),
            source,
            scopes: vec![HashMap::new()], // start with empty global scope
            ..Default::default()
        }
    }

    /// Adds a new diagnostic with the `Hint` level.
    fn hint(&mut self, message: impl Into<Cow<'static, str>>, span: Span<'src>) {
        self.diagnostics.push(Diagnostic::new(
            DiagnosticLevel::Hint,
            message,
            vec![],
            span,
            self.source,
        ))
    }

    /// Adds a new diagnostic with the `Info` level.
    fn info(
        &mut self,
        message: impl Into<Cow<'static, str>>,
        notes: Vec<Cow<'static, str>>,
        span: Span<'src>,
    ) {
        self.diagnostics.push(Diagnostic::new(
            DiagnosticLevel::Info,
            message,
            notes,
            span,
            self.source,
        ))
    }

    /// Adds a new diagnostic with the `Warning` level.
    fn warn(
        &mut self,
        message: impl Into<Cow<'static, str>>,
        notes: Vec<Cow<'static, str>>,
        span: Span<'src>,
    ) {
        self.diagnostics.push(Diagnostic::new(
            DiagnosticLevel::Warning,
            message,
            notes,
            span,
            self.source,
        ))
    }

    /// Adds a new diagnostic with the `Error` level using the specified error kind.
    fn error(
        &mut self,
        kind: ErrorKind,
        message: impl Into<Cow<'static, str>>,
        notes: Vec<Cow<'static, str>>,
        span: Span<'src>,
    ) {
        self.diagnostics.push(Diagnostic::new(
            DiagnosticLevel::Error(kind),
            message,
            notes,
            span,
            self.source,
        ))
    }

    /// Analyzes a parsed AST and returns an analyzed AST whilst emitting diagnostics.
    pub fn analyze(
        mut self,
        program: Program<'src>,
    ) -> Result<(AnalyzedProgram<'src>, Vec<Diagnostic>), Vec<Diagnostic>> {
        // add all function signatures first
        for func in &program.functions {
            // check for duplicate function names
            if let Some(prev_def) = self.functions.get(func.name.inner) {
                let prev_def_span = prev_def.ident.span;
                self.error(
                    ErrorKind::Semantic,
                    format!("duplicate function definition `{}`", func.name.inner),
                    vec![],
                    func.name.span,
                );
                self.hint(
                    format!("function `{}` previously defined here", func.name.inner),
                    prev_def_span,
                );
            }
            self.functions.insert(
                func.name.inner,
                Function {
                    ident: func.name.clone(),
                    params: func.params.clone(),
                    return_type: func.return_type.clone(),
                    used: false,
                },
            );
        }

        // analyze global let stmts
        // `self.global(node)` has side effects that have to happen here
        #[allow(clippy::needless_collect)]
        let globals: Vec<AnalyzedLetStmt> = program
            .globals
            .into_iter()
            .map(|node| self.global(node))
            .collect();

        // then analyze each function body
        let mut functions = vec![];
        let mut main_fn = None;
        for func in program.functions {
            let func = self.function_definition(func);
            match func.name {
                "main" => {
                    main_fn = Some(func.block);
                }
                _ => functions.push(func),
            }
        }

        // pop the global scope
        let (unused_globals, non_mut_globals) = self.pop_scope();
        let globals: Vec<AnalyzedLetStmt> = globals
            .into_iter()
            .map(|g| AnalyzedLetStmt {
                used: !unused_globals.contains(&g.name),
                mutable: g.mutable && !non_mut_globals.contains(&g.name),
                ..g
            })
            .collect();

        // check if there are any unused functions
        let unused_funcs: Vec<_> = self
            .functions
            .values()
            .filter(|func| {
                func.ident.inner != "main" && !func.ident.inner.starts_with('_') && !func.used
            })
            .map(|func| {
                // set used = false in tree
                functions
                    .iter_mut()
                    .find(|func_def| func_def.name == func.ident.inner)
                    .expect("every unused function is defined")
                    .used = false;

                func.ident.clone()
            })
            .collect();

        // add warnings to unused functions
        for ident in unused_funcs {
            self.warn(
                format!("function `{}` is never called", ident.inner),
                vec![format!(
                    "if this is intentional, change the name to `_{}` to hide this warning",
                    ident.inner,
                )
                .into()],
                ident.span,
            )
        }

        match main_fn {
            Some(main_fn) => Ok((
                AnalyzedProgram {
                    globals,
                    functions,
                    main_fn,
                    used_builtins: self.used_builtins,
                },
                self.diagnostics,
            )),
            None => {
                self.error(
                    ErrorKind::Semantic,
                    "missing `main` function",
                    vec![
                        "the `main` function can be implemented like this: `fn main() { ... }`"
                            .into(),
                    ],
                    // empty span including filename
                    program.span.start.until(program.span.start),
                );
                Err(self.diagnostics)
            }
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Removes the current scope of the function and checks whether the
    /// variables in the scope have been used and/or mutated.
    /// Returns the names of variables which are unused and those which do
    /// not need to be mutable.
    fn pop_scope(&mut self) -> (Vec<&'src str>, Vec<&'src str>) {
        // consume / drop the scope
        let scope = self.scopes.pop().expect("is only called after a scope");

        let mut unused = vec![];
        let mut non_mut = vec![];

        // analyze its values for their use
        for (name, var) in scope {
            // allow unused values if they start with `_`
            if !name.starts_with('_') && !var.used {
                unused.push(name);
                self.warn(
                    format!("unused variable `{}`", name),
                    vec![format!(
                        "if this is intentional, change the name to `_{name}` to hide this warning"
                    )
                    .into()],
                    var.span,
                );
            } else if var.mutable && !var.mutated {
                non_mut.push(name);
                self.info(
                    format!("variable `{name}` does not need to be mutable"),
                    vec![],
                    var.span,
                );
            }
        }

        (unused, non_mut)
    }

    // Returns a mutable reference to the current scope
    fn scope_mut(&mut self) -> &mut HashMap<&'src str, Variable<'src>> {
        self.scopes.last_mut().expect("only called in scopes")
    }

    fn warn_unreachable(
        &mut self,
        unreachable_span: Span<'src>,
        causing_span: Span<'src>,
        expr: bool,
    ) {
        self.warn(
            match expr {
                true => "unreachable expression",
                false => "unreachable statement",
            },
            vec![],
            unreachable_span,
        );
        self.hint(
            "any code following this expression is unreachable",
            causing_span,
        );
    }

    fn global(&mut self, node: LetStmt<'src>) -> AnalyzedLetStmt<'src> {
        // analyze the right hand side first
        let expr_span = node.expr.span();
        let expr = self.expression(node.expr);

        // check if the expression is constant
        if !expr.constant() {
            self.error(
                ErrorKind::Semantic,
                "global initializer is not constant",
                vec!["global variables must have a constant initializer".into()],
                expr_span,
            );
        }

        // check if the optional type conflicts with the rhs
        if let Some(declared) = &node.type_ {
            if declared.inner != expr.result_type()
                && !matches!(expr.result_type(), Type::Unknown | Type::Never)
            {
                self.error(
                    ErrorKind::Type,
                    format!(
                        "mismatched types: expected `{}`, found `{}`",
                        declared.inner,
                        expr.result_type(),
                    ),
                    vec![],
                    expr_span,
                );
                self.hint("expected due to this", declared.span);
            }
        }

        // do not allow duplicate globals
        if let Some(prev) = self.scopes[0].get(node.name.inner) {
            let prev_span = prev.span;
            self.error(
                ErrorKind::Semantic,
                format!("duplicate definition of global `{}`", node.name.inner),
                vec![],
                node.name.span,
            );
            self.hint(
                format!("previous definition of global `{}` here", node.name.inner),
                prev_span,
            );
        } else {
            self.scopes[0].insert(
                node.name.inner,
                Variable {
                    // use `{unknown}` type for non-constant globals to prevent further misleading
                    // warnings
                    type_: match expr.constant() {
                        true => node.type_.map_or(expr.result_type(), |type_| type_.inner),
                        false => Type::Unknown,
                    },
                    span: node.name.span,
                    used: false,
                    mutable: node.mutable,
                    mutated: false,
                },
            );
        }

        AnalyzedLetStmt {
            name: node.name.inner,
            expr,
            mutable: node.mutable,
            used: false,
        }
    }

    fn function_definition(
        &mut self,
        node: FunctionDefinition<'src>,
    ) -> AnalyzedFunctionDefinition<'src> {
        // set the function name
        self.curr_func_name = node.name.inner;

        if node.name.inner == "main" {
            // the main function must have 0 parameters
            if !node.params.inner.is_empty() {
                self.error(
                    ErrorKind::Semantic,
                    format!(
                        "the `main` function must have 0 parameters, however {} {} defined",
                        node.params.inner.len(),
                        match node.params.inner.len() {
                            1 => "is",
                            _ => "are",
                        },
                    ),
                    vec!["remove the parameters: `fn main() { ... }`".into()],
                    node.params.span,
                )
            }

            // the main function must return `()`
            if let Some(return_type) = node.return_type.inner {
                if return_type != Type::Unit {
                    self.error(
                        ErrorKind::Semantic,
                        format!(
                            "the `main` function's return type must be `()`, but is declared as `{}`",
                            return_type,
                        ),
                        vec!["remove the return type: `fn main() { ... }`".into()],
                        node.return_type.span,
                    )
                }
            }
        }

        // info for explicit unit return type
        if node.return_type.inner == Some(Type::Unit) {
            self.info(
                "unnecessary explicit unit return type",
                vec![
                    "functions implicitly return `()` by default".into(),
                    format!(
                        "remove the explicit type: `fn {}(...) {{ ... }}`",
                        node.name.inner,
                    )
                    .into(),
                ],
                node.return_type.span,
            );
        }

        // push a new scope for the new function
        self.push_scope();

        // check the function parameters
        let mut params = vec![];
        let mut param_names = HashSet::new();

        // only analyze parameters if this is not the main function
        for param in node.params.inner {
            // check for duplicate function parameters
            if !param_names.insert(param.name.inner) && node.name.inner != "main" {
                self.error(
                    ErrorKind::Semantic,
                    format!("duplicate parameter name `{}`", param.name.inner),
                    vec![],
                    param.name.span,
                );
            }
            self.scope_mut().insert(
                param.name.inner,
                Variable {
                    type_: param.type_.inner,
                    span: param.name.span,
                    used: false,
                    mutable: param.mutable,
                    mutated: false,
                },
            );
            params.push(AnalyzedParameter {
                mutable: param.mutable,
                name: param.name.inner,
                type_: param.type_.inner,
            });
        }

        // analyze the function body
        let block_result_span = node.block.result_span();
        let block = self.block(node.block, false);

        // check that the block results in the expected type
        if block.result_type != node.return_type.inner.unwrap_or(Type::Unit)
            // unknown and never types are tolerated
            && !matches!(block.result_type, Type::Unknown | Type::Never)
        {
            self.error(
                ErrorKind::Type,
                format!(
                    "mismatched types: expected `{}`, found `{}`",
                    node.return_type.inner.unwrap_or(Type::Unit),
                    block.result_type,
                ),
                match node.return_type.inner.is_some() {
                    true => vec![],
                    false => vec![format!(
                        "specify a function return type like this: `fn {}(...) -> {} {{ ... }}`",
                        node.name.inner, block.result_type,
                    )
                    .into()],
                },
                block_result_span,
            );
            self.hint(
                match node.return_type.inner.is_some() {
                    true => "function return type defined here",
                    false => "no explicit return type specified",
                },
                node.return_type.span,
            );
        }

        // drop the scope when finished
        self.pop_scope();

        // issue a warning if there are more than 6 parameters
        if params.len() > 6 {
            self.warn(
                "function takes more than 6 parameters".to_string(),
                vec!["using more than 6 parameters will be slower on some backends".into()],
                node.params.span,
            )
        }

        AnalyzedFunctionDefinition {
            used: true, // is modified in Self::analyze()
            name: node.name.inner,
            params,
            return_type: node.return_type.inner.unwrap_or(Type::Unit),
            block,
        }
    }

    fn block(&mut self, node: Block<'src>, new_scope: bool) -> AnalyzedBlock<'src> {
        if new_scope {
            self.push_scope();
        }

        let mut stmts = vec![];

        let mut never_type_span = None;
        let mut warned_unreachable = false;

        for stmt in node.stmts {
            if let Some(span) = never_type_span {
                if !warned_unreachable {
                    self.warn("unreachable statement", vec![], stmt.span());
                    self.hint("any code following this statement is unreachable", span);
                    warned_unreachable = true;
                }
            }
            let stmt_span = stmt.span();
            if let Some(stmt) = self.statement(stmt) {
                if stmt.result_type() == Type::Never {
                    never_type_span = Some(stmt_span);
                }
                stmts.push(stmt);
            }
        }

        // possibly mark trailing expression as unreachable
        if let (Some(expr), Some(span), false) = (&node.expr, never_type_span, warned_unreachable) {
            self.warn("unreachable expression", vec![], expr.span());
            self.hint("any code following this statement is unreachable", span);
        }

        // analyze the expression
        let expr = node.expr.map(|expr| self.expression(expr));

        // result type is `!` when any statement had type `!`, otherwise the type of the expr
        let result_type = match never_type_span {
            Some(_) => Type::Never,
            None => expr.as_ref().map_or(Type::Unit, |expr| expr.result_type()),
        };

        if new_scope {
            self.pop_scope();
        }

        AnalyzedBlock {
            result_type,
            stmts,
            expr,
        }
    }

    /// Analyzes a [`Statement`].
    /// Can return [`None`] if the statement is a `while` loop which never loops.
    fn statement(&mut self, node: Statement<'src>) -> Option<AnalyzedStatement<'src>> {
        Some(match node {
            Statement::Let(node) => self.let_stmt(node),
            Statement::Return(node) => self.return_stmt(node),
            Statement::Loop(node) => self.loop_stmt(node),
            Statement::While(node) => return self.while_stmt(node),
            Statement::For(node) => self.for_stmt(node),
            Statement::Break(node) => self.break_stmt(node),
            Statement::Continue(node) => self.continue_stmt(node),
            Statement::Expr(node) => AnalyzedStatement::Expr(self.expression(node.expr)),
        })
    }

    fn let_stmt(&mut self, node: LetStmt<'src>) -> AnalyzedStatement<'src> {
        // save the expression's span for later use
        let expr_span = node.expr.span();

        // analyze the right hand side first
        let expr = self.expression(node.expr);

        // check if the optional type conflicts with the rhs
        if let Some(declared) = &node.type_ {
            if declared.inner != expr.result_type()
                && !matches!(expr.result_type(), Type::Unknown | Type::Never)
            {
                self.error(
                    ErrorKind::Type,
                    format!(
                        "mismatched types: expected `{}`, found `{}`",
                        declared.inner,
                        expr.result_type(),
                    ),
                    vec![],
                    expr_span,
                );
                self.hint("expected due to this", declared.span);
            }
        }

        // warn unreachable if never type
        if expr.result_type() == Type::Never {
            self.warn_unreachable(node.span, expr_span, false);
        }

        // insert and do additional checks if variable is shadowed
        if let Some(shadowed) = self.scope_mut().insert(
            node.name.inner,
            Variable {
                type_: match node.type_.map_or(expr.result_type(), |type_| type_.inner) {
                    // map `!` to `{unknown}` to prevent misleading warnings
                    Type::Never => Type::Unknown,
                    type_ => type_,
                },
                span: node.name.span,
                used: false,
                mutable: node.mutable,
                mutated: false,
            },
        ) {
            // a previous variable is shadowed by this declaration, analyze its use
            if !shadowed.used && !node.name.inner.starts_with('_') {
                self.warn(
                    format!("unused variable `{}`", node.name.inner),
                    vec![format!(
                        "if this is intentional, change the name to `_{}` to hide this warning",
                        node.name.inner
                    )
                    .into()],
                    shadowed.span,
                );
                self.hint(
                    format!("variable `{}` shadowed here", node.name.inner),
                    node.name.span,
                );
            } else if shadowed.mutable && !shadowed.mutated {
                self.info(
                    format!("variable `{}` does not need to be mutable", node.name.inner),
                    vec![],
                    shadowed.span,
                );
            }
        }

        AnalyzedStatement::Let(AnalyzedLetStmt {
            name: node.name.inner,
            expr,
            mutable: node.mutable,
            used: true,
        })
    }

    fn return_stmt(&mut self, node: ReturnStmt<'src>) -> AnalyzedStatement<'src> {
        // if there is an expression, visit it
        let expr_span = node.expr.as_ref().map(|expr| expr.span());
        let expr = node.expr.map(|expr| self.expression(expr));

        // get the return type based on the expr (Unit as fallback)
        let expr_type = expr.as_ref().map_or(Type::Unit, |expr| expr.result_type());

        if expr_type == Type::Never {
            self.warn_unreachable(
                node.span,
                expr_span.expect("the never type was caused by an expression"),
                false,
            );
        }

        let curr_fn = &self.functions[self.curr_func_name];

        // test if the return type is correct
        if curr_fn.return_type.inner.unwrap_or(Type::Unit) != expr_type
            // unknown and never types are tolerated
            && !matches!(expr_type, Type::Unknown | Type::Never)
        {
            let fn_type_span = curr_fn.return_type.span;
            let fn_type_explicit = curr_fn.return_type.inner.is_some();

            self.error(
                ErrorKind::Type,
                format!(
                    "mismatched types: expected `{}`, found `{}`",
                    curr_fn.return_type.inner.unwrap_or(Type::Unit),
                    expr_type
                ),
                vec![],
                node.span,
            );
            self.hint(
                match fn_type_explicit {
                    true => "function return type defined here",
                    false => "no explicit return type specified",
                },
                fn_type_span,
            );
        }

        AnalyzedStatement::Return(expr)
    }

    fn loop_stmt(&mut self, node: LoopStmt<'src>) -> AnalyzedStatement<'src> {
        let old_loop_is_terminated = self.current_loop_is_terminated;

        self.loop_count += 1;
        let block_result_span = node.block.result_span();
        let block = self.block(node.block, true);
        self.loop_count -= 1;

        if !matches!(block.result_type, Type::Unit | Type::Never | Type::Unknown) {
            self.error(
                ErrorKind::Type,
                format!(
                    "loop-statement requires a block of type `()` or `!`, found `{}`",
                    block.result_type
                ),
                vec![],
                block_result_span,
            );
        }

        // restore loop termination count
        let never_terminates = !self.current_loop_is_terminated;
        self.current_loop_is_terminated = old_loop_is_terminated;

        AnalyzedStatement::Loop(AnalyzedLoopStmt {
            block,
            never_terminates,
        })
    }

    /// Analyzes a [`WhileStmt`].
    /// Will return [`None`] if the loop never iterates (condition is constant `false`)
    /// Can also return an [`AnalyzedLoopStmt`] if the expression is constant `true`.
    fn while_stmt(&mut self, node: WhileStmt<'src>) -> Option<AnalyzedStatement<'src>> {
        let mut condition_is_const_true = false;
        let mut never_loops = false;

        let cond_span = node.cond.span();
        let cond = self.expression(node.cond);

        // check that the condition is of type bool
        if !matches!(
            cond.result_type(),
            Type::Bool(0) | Type::Never | Type::Unknown
        ) {
            self.error(
                ErrorKind::Type,
                format!(
                    "expected value of type `bool`, found `{}`",
                    cond.result_type()
                ),
                vec!["a condition must have the type `bool`".into()],
                cond_span,
            )
        } else {
            // check that the condition is non-constant
            if cond.constant() {
                let cond_val = match cond {
                    AnalyzedExpression::Bool(true) => {
                        condition_is_const_true = true;
                        true
                    }
                    AnalyzedExpression::Bool(false) => {
                        never_loops = true;
                        false
                    }
                    _ => unreachable!("type is checked above and expr is constant"),
                };
                self.warn(
                    format!("redundant while-statement: condition is always {cond_val}"),
                    match cond_val {
                        true => vec!["for unconditional loops, use a loop-statement".into()],
                        false => vec![
                            "since the condition is `false`, the loop will never iterate".into(),
                        ],
                    },
                    cond_span,
                )
            }
        }

        let old_loop_is_terminated = self.current_loop_is_terminated;

        self.loop_count += 1;
        let block_result_span = node.block.result_span();
        let body_is_empty = node.block.stmts.is_empty() && node.block.expr.is_none();
        let block = self.block(node.block, true);
        self.loop_count -= 1;

        if body_is_empty {
            self.warn(
                "empty loop body",
                vec!["empty loop wastes CPU cycles".into()],
                node.span,
            )
        }

        if !matches!(block.result_type, Type::Unit | Type::Never | Type::Unknown) {
            self.error(
                ErrorKind::Type,
                format!(
                    "while-statement requires a block of type `()` or `!`, found `{}`",
                    block.result_type
                ),
                vec![],
                block_result_span,
            );
        }

        // restore loop termination count
        let never_terminates = condition_is_const_true && !self.current_loop_is_terminated;
        self.current_loop_is_terminated = old_loop_is_terminated;

        match (never_loops, condition_is_const_true) {
            // if the condition is always `false`, return nothing
            (true, _) => None,
            // if the condition is always `true`, return an `AnalyzedLoopStmt`
            (false, true) => Some(AnalyzedStatement::Loop(AnalyzedLoopStmt {
                block,
                never_terminates,
            })),
            // otherwise, return an `AnalyzedWhileStmt`
            (false, false) => Some(AnalyzedStatement::While(AnalyzedWhileStmt {
                cond,
                block,
                never_terminates,
            })),
        }
    }

    fn for_stmt(&mut self, node: ForStmt<'src>) -> AnalyzedStatement<'src> {
        let mut never_terminates = false;

        // push the scope here so that the initializer is in the new scope
        self.push_scope();

        // analyze the initializer
        let initializer = self.expression(node.initializer);
        self.scope_mut().insert(
            node.ident.inner,
            Variable {
                type_: initializer.result_type(),
                span: node.ident.span,
                used: false,
                mutable: true,
                // always set mutated = true, even if it is not mutated, to prevent weird warnings
                mutated: true,
            },
        );

        // check that the condition is of type bool
        let cond_span = node.cond.span();
        let cond = self.expression(node.cond);

        if !matches!(
            cond.result_type(),
            Type::Bool(0) | Type::Never | Type::Unknown
        ) {
            self.error(
                ErrorKind::Type,
                format!(
                    "expected value of type `bool`, found `{}`",
                    cond.result_type()
                ),
                vec!["a condition must have the type `bool`".into()],
                cond_span,
            )
        } else {
            // check that the condition is non-constant
            if cond.constant() {
                let cond_val = match cond {
                    AnalyzedExpression::Bool(true) => {
                        never_terminates = true;
                        true
                    }
                    AnalyzedExpression::Bool(false) => false,
                    _ => unreachable!("type is checked above and expr is constant"),
                };
                self.warn(
                    format!("redundant for-statement: condition is always {cond_val}",),
                    match cond_val {
                        true => vec!["for unconditional loops, use a loop-statement".into()],
                        false => vec![
                            "since the condition is `false`, the loop will never iterate".into(),
                        ],
                    },
                    cond_span,
                )
            }
        }

        // check that the update expr results in `()`, `!` or `{unknown}`
        let upd_span = node.update.span();
        let update = self.expression(node.update);
        if !matches!(
            update.result_type(),
            Type::Unit | Type::Never | Type::Unknown
        ) {
            self.error(
                ErrorKind::Type,
                format!(
                    "expected value of type `()`, found `{}`",
                    update.result_type()
                ),
                vec!["an update expression must have the type `()` or `!`".into()],
                upd_span,
            )
        }

        // save the old status of loop termination
        let old_loop_is_terminated = self.current_loop_is_terminated;

        self.loop_count += 1;
        let block_result_span = node.block.result_span();
        let block = self.block(node.block, false);
        self.pop_scope();
        self.loop_count -= 1;

        // restore loop termination count
        let never_terminates = never_terminates && !self.current_loop_is_terminated;
        self.current_loop_is_terminated = old_loop_is_terminated;

        if !matches!(block.result_type, Type::Unit | Type::Never | Type::Unknown) {
            self.error(
                ErrorKind::Type,
                format!(
                    "for-statement requires a block of type `()` or `!`, found `{}`",
                    block.result_type
                ),
                vec![],
                block_result_span,
            );
        }

        AnalyzedStatement::For(AnalyzedForStmt {
            ident: node.ident.inner,
            initializer,
            cond,
            update,
            block,
            never_terminates,
        })
    }

    fn break_stmt(&mut self, node: BreakStmt<'src>) -> AnalyzedStatement<'src> {
        if self.loop_count == 0 {
            self.error(
                ErrorKind::Semantic,
                "`break` outside of loop",
                vec![],
                node.span,
            );
        }
        self.current_loop_is_terminated = true;
        AnalyzedStatement::Break
    }

    fn continue_stmt(&mut self, node: ContinueStmt<'src>) -> AnalyzedStatement<'src> {
        if self.loop_count == 0 {
            self.error(
                ErrorKind::Semantic,
                "`continue` outside of loop",
                vec![],
                node.span,
            );
        }
        AnalyzedStatement::Continue
    }

    fn expression(&mut self, node: Expression<'src>) -> AnalyzedExpression<'src> {
        let res = match node {
            Expression::Int(node) => AnalyzedExpression::Int(node.inner),
            Expression::Float(node) => AnalyzedExpression::Float(node.inner),
            Expression::Bool(node) => AnalyzedExpression::Bool(node.inner),
            Expression::Char(node) => {
                if node.inner > 0x7f {
                    self.error(
                        ErrorKind::Type,
                        "char literal out of range".to_string(),
                        vec![
                            format!("allowed range is `0x00..=0x7f`, got `0x{:x}`", node.inner)
                                .into(),
                        ],
                        node.span,
                    )
                }
                AnalyzedExpression::Char(node.inner)
            }
            Expression::Ident(node) => self.ident_expr(node),
            Expression::Prefix(node) => self.prefix_expr(*node),
            Expression::Infix(node) => self.infix_expr(*node),
            Expression::Assign(node) => self.assign_expr(*node),
            Expression::Call(node) => self.call_expr(*node),
            Expression::Cast(node) => self.cast_expr(*node),
            Expression::If(node) => self.if_expr(*node),
            Expression::Block(node) => self.block_expr(*node),
            Expression::Grouped(node) => {
                let expr = self.expression(*node.inner);
                match expr.as_constant() {
                    Some(expr) => expr,
                    None => AnalyzedExpression::Grouped(expr.into()),
                }
            }
        };

        // if this is a `!` expression, count it like a loop termination
        if res.result_type() == Type::Never {
            self.current_loop_is_terminated = true;
        }

        res
    }

    fn block_expr(&mut self, node: Block<'src>) -> AnalyzedExpression<'src> {
        let block = self.block(node, true);

        match Self::eval_block(&block) {
            Some(expr) => expr,
            None => AnalyzedExpression::Block(block.into()),
        }
    }

    fn eval_block(block: &AnalyzedBlock<'src>) -> Option<AnalyzedExpression<'src>> {
        if block.stmts.iter().all(|stmt| stmt.constant()) {
            if let Some(expr) = block.expr.as_ref().and_then(|expr| expr.as_constant()) {
                return Some(expr);
            }
        }
        None
    }

    fn if_expr(&mut self, node: IfExpr<'src>) -> AnalyzedExpression<'src> {
        let cond_span = node.cond.span();
        let cond = self.expression(node.cond);

        // check that the condition is of type bool
        if !matches!(
            cond.result_type(),
            Type::Bool(0) | Type::Never | Type::Unknown
        ) {
            self.error(
                ErrorKind::Type,
                format!(
                    "expected value of type `bool`, found `{}`",
                    cond.result_type()
                ),
                vec!["a condition must have the type `bool`".into()],
                cond_span,
            )
        } else {
            // check that the condition is non-constant
            if cond.constant() {
                self.warn(
                    format!(
                        "redundant if-expression: condition is always {}",
                        match cond {
                            AnalyzedExpression::Bool(true) => "true",
                            AnalyzedExpression::Bool(false) => "false",
                            _ => unreachable!("type is checked above and expr is constant"),
                        }
                    ),
                    vec![],
                    cond_span,
                )
            }
        }

        // analyze then_block
        let then_result_span = node.then_block.result_span();
        let then_block = self.block(node.then_block, true);

        // analyze else_block if it exists
        let result_type;
        let else_block = match node.else_block {
            Some(else_block) => {
                let else_result_span = else_block.result_span();
                let else_block = self.block(else_block, true);

                // check type equality of the `then` and `else` branches
                result_type = match (then_block.result_type, else_block.result_type) {
                    // unknown when any branch is unknown
                    (Type::Unknown, _) | (_, Type::Unknown) => Type::Unknown,
                    // never when both branches are never
                    (Type::Never, Type::Never) => Type::Never,
                    // the type of the non-never branch when one branch is never
                    (type_, Type::Never) | (Type::Never, type_) => type_,
                    // the then_type when both branches have the same type
                    (then_type, else_type) if then_type == else_type => then_type,
                    // unknown and error otherwise
                    _ => {
                        self.error(
                            ErrorKind::Type,
                            format!(
                                "mismatched types: expected `{}`, found `{}`",
                                then_block.result_type, else_block.result_type
                            ),
                            vec!["the `if` and `else` branches must result in the same type".into()],
                            else_result_span,
                        );
                        self.hint("expected due to this", then_result_span);
                        Type::Unknown
                    }
                };

                Some(else_block)
            }
            None => {
                result_type = match then_block.result_type {
                    Type::Unknown => Type::Unknown,
                    Type::Unit | Type::Never => Type::Unit,
                    _ => {
                        self.error(
                            ErrorKind::Type,
                            format!(
                                "mismatched types: missing else branch with `{}` result type",
                                then_block.result_type
                            ),
                            vec![format!("the `if` branch results in `{}`, therefore an else branch was expected", then_block.result_type).into()],
                            node.span,
                        );
                        Type::Unknown
                    }
                };

                None
            }
        };

        // evaluate constant if-exprs
        match (
            cond.as_constant(),
            Self::eval_block(&then_block),
            else_block.as_ref().and_then(Self::eval_block),
        ) {
            (Some(AnalyzedExpression::Bool(true)), Some(val), Some(_)) => return val,
            (Some(AnalyzedExpression::Bool(false)), Some(_), Some(val)) => return val,
            _ => {}
        }

        AnalyzedExpression::If(
            AnalyzedIfExpr {
                result_type,
                cond,
                then_block,
                else_block,
            }
            .into(),
        )
    }

    /// Searches all scopes for the requested variable.
    /// Starts at the current scope (last) and works its way down to the global scope (first).
    fn ident_expr(&mut self, node: Spanned<'src, &'src str>) -> AnalyzedExpression<'src> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(var) = scope.get_mut(node.inner) {
                var.used = true;
                return AnalyzedExpression::Ident(AnalyzedIdentExpr {
                    result_type: var.type_,
                    ident: node.inner,
                });
            };
        }

        // ignore empty identifiers (cannot be created by users)
        if !node.inner.is_empty() {
            self.error(
                ErrorKind::Reference,
                format!("use of undeclared variable `{}`", node.inner),
                vec![],
                node.span,
            );
        }

        AnalyzedExpression::Ident(AnalyzedIdentExpr {
            result_type: Type::Unknown,
            ident: node.inner,
        })
    }

    fn prefix_expr(&mut self, node: PrefixExpr<'src>) -> AnalyzedExpression<'src> {
        let expr_span = node.expr.span();
        let expr = self.expression(node.expr);

        let result_type = match node.op {
            PrefixOp::Not => match expr.result_type() {
                Type::Bool(0) => Type::Bool(0),
                Type::Int(0) => Type::Int(0),
                Type::Unknown => Type::Unknown,
                Type::Never => {
                    self.warn_unreachable(node.span, expr_span, true);
                    Type::Never
                }
                _ => {
                    self.error(
                        ErrorKind::Type,
                        format!(
                            "prefix operator `!` does not allow values of type `{}`",
                            expr.result_type()
                        ),
                        vec![],
                        node.span,
                    );
                    Type::Unknown
                }
            },
            PrefixOp::Neg => match expr.result_type() {
                Type::Int(0) => Type::Int(0),
                Type::Float(0) => Type::Float(0),
                Type::Unknown => Type::Unknown,
                Type::Never => {
                    self.warn_unreachable(node.span, expr_span, true);
                    Type::Never
                }
                _ => {
                    self.error(
                        ErrorKind::Type,
                        format!(
                            "prefix operator `-` does not allow values of type `{}`",
                            expr.result_type()
                        ),
                        vec![],
                        node.span,
                    );
                    Type::Unknown
                }
            },
            PrefixOp::Ref => match &expr {
                AnalyzedExpression::Ident(ident) => match ident.result_type.add_ref() {
                    Some(res) => {
                        let mut var = self
                            .scopes
                            .iter_mut()
                            .rev()
                            .find_map(|s| s.get_mut(ident.ident))
                            .expect("variable references are valid here");

                        // references (`&`) count as mutable variable accesses
                        var.mutated = true;

                        // validate that the variable is mutable
                        let var_span = var.span;
                        if !var.mutable {
                            self.error(
                                ErrorKind::Semantic,
                                format!("cannot reference immutable variable `{}`", ident.ident),
                                vec!["only mutable variables can be referenced".into()],
                                node.span,
                            );
                            self.hint("variable not declared as `mut`", var_span);
                        }

                        res
                    }
                    None if ident.result_type == Type::Unknown => Type::Unknown,
                    None => {
                        self.error(
                            ErrorKind::Type,
                            format!("cannot reference a value of type `{}`", ident.result_type),
                            vec![],
                            node.span,
                        );
                        Type::Unknown
                    }
                },
                _ => unreachable!("parser guarantees that only identifiers are referenced"),
            },
            PrefixOp::Deref => match &expr {
                // TODO: improve this
                AnalyzedExpression::Ident(ident) => match ident.result_type.sub_deref() {
                    Some(res) => res,
                    None => {
                        self.error(
                            ErrorKind::Type,
                            format!("cannot dereference a value of type `{}`", ident.result_type),
                            vec!["only pointers `*` can be dereferenced".into()],
                            node.span,
                        );
                        dbg!(ident);
                        Type::Unknown
                    }
                },
                AnalyzedExpression::Prefix(expr) => match expr.result_type.sub_deref() {
                    Some(res) => res,
                    None => {
                        self.error(
                            ErrorKind::Type,
                            format!("cannot dereference a value of type `{}`", expr.result_type),
                            vec!["only pointers `*` can be dereferenced".into()],
                            node.span,
                        );
                        Type::Unknown
                    }
                },
                _ => unreachable!("can only dereference identifiers or prefix expressions"),
            },
        };

        // evaluate constant expressions
        match (&expr, node.op) {
            (AnalyzedExpression::Int(num), PrefixOp::Not) => return AnalyzedExpression::Int(!num),
            (AnalyzedExpression::Int(num), PrefixOp::Neg) => {
                return AnalyzedExpression::Int(num.wrapping_neg())
            }
            (AnalyzedExpression::Float(num), PrefixOp::Neg) => {
                return AnalyzedExpression::Float(-num)
            }
            (AnalyzedExpression::Bool(bool), PrefixOp::Not) => {
                return AnalyzedExpression::Bool(!bool)
            }
            _ => {}
        }

        AnalyzedExpression::Prefix(
            AnalyzedPrefixExpr {
                result_type,
                op: node.op,
                expr,
            }
            .into(),
        )
    }

    fn infix_expr(&mut self, node: InfixExpr<'src>) -> AnalyzedExpression<'src> {
        let lhs_span = node.lhs.span();
        let rhs_span = node.rhs.span();
        let lhs = self.expression(node.lhs);
        let rhs = self.expression(node.rhs);

        let allowed_types: &[Type];
        let mut override_result_type = None;
        let mut inherits_never_type = true;
        match node.op {
            InfixOp::Plus | InfixOp::Minus => {
                allowed_types = &[Type::Int(0), Type::Char(0), Type::Float(0)];
            }
            InfixOp::Mul | InfixOp::Div => {
                allowed_types = &[Type::Int(0), Type::Float(0)];
            }
            InfixOp::Lt | InfixOp::Gt | InfixOp::Lte | InfixOp::Gte => {
                allowed_types = &[Type::Int(0), Type::Char(0), Type::Float(0)];
                override_result_type = Some(Type::Bool(0));
            }
            InfixOp::Rem | InfixOp::Shl | InfixOp::Shr | InfixOp::Pow => {
                allowed_types = &[Type::Int(0)];
            }
            InfixOp::Eq | InfixOp::Neq => {
                allowed_types = &[Type::Int(0), Type::Float(0), Type::Bool(0), Type::Char(0)];
                override_result_type = Some(Type::Bool(0));
            }
            InfixOp::BitOr | InfixOp::BitAnd | InfixOp::BitXor => {
                allowed_types = &[Type::Int(0), Type::Bool(0)];
            }
            InfixOp::And | InfixOp::Or => {
                allowed_types = &[Type::Bool(0)];
                inherits_never_type = false;
            }
        }

        let result_type = match (lhs.result_type(), rhs.result_type()) {
            (Type::Unknown, _) | (_, Type::Unknown) => Type::Unknown,
            (Type::Never, Type::Never) => {
                self.warn_unreachable(node.span, lhs_span, true);
                self.warn_unreachable(node.span, rhs_span, true);
                Type::Never
            }
            (Type::Never, _) if inherits_never_type => {
                self.warn_unreachable(node.span, lhs_span, true);
                Type::Never
            }
            (_, Type::Never) if inherits_never_type => {
                self.warn_unreachable(node.span, rhs_span, true);
                Type::Never
            }
            (Type::Never, _) => rhs.result_type(),
            (_, Type::Never) => lhs.result_type(),
            (left, right) if left == right && allowed_types.contains(&left) => {
                override_result_type.unwrap_or(left)
            }
            (left, right) if left != right => {
                self.error(
                    ErrorKind::Type,
                    format!(
                        "infix expressions require equal types on both sides, got `{left}` and `{right}`"
                    ),
                    vec![],
                    node.span,
                );
                Type::Unknown
            }
            (type_, _) => {
                self.error(
                    ErrorKind::Type,
                    format!(
                        "infix operator `{}` does not allow values of type `{type_}`",
                        node.op
                    ),
                    vec![],
                    node.span,
                );
                Type::Unknown
            }
        };

        // evaluate constant expressions
        match (&lhs, &rhs) {
            (AnalyzedExpression::Char(left), AnalyzedExpression::Char(right)) => match node.op {
                InfixOp::Plus => return AnalyzedExpression::Char(left.wrapping_add(*right) & 0x7f),
                InfixOp::Minus => {
                    return AnalyzedExpression::Char(left.wrapping_sub(*right) & 0x7f)
                }
                InfixOp::Eq => return AnalyzedExpression::Bool(left == right),
                InfixOp::Neq => return AnalyzedExpression::Bool(left != right),
                InfixOp::Lt => return AnalyzedExpression::Bool(left < right),
                InfixOp::Lte => return AnalyzedExpression::Bool(left <= right),
                InfixOp::Gt => return AnalyzedExpression::Bool(left > right),
                InfixOp::Gte => return AnalyzedExpression::Bool(left >= right),
                _ => {}
            },
            (AnalyzedExpression::Int(left), AnalyzedExpression::Int(right)) => match node.op {
                InfixOp::Plus => return AnalyzedExpression::Int(left.wrapping_add(*right)),
                InfixOp::Minus => return AnalyzedExpression::Int(left.wrapping_sub(*right)),
                InfixOp::Mul => return AnalyzedExpression::Int(left.wrapping_mul(*right)),
                InfixOp::Div if *right == 0 => self.error(
                    ErrorKind::Semantic,
                    format!("cannot divide {left} by 0"),
                    vec!["division by 0 is undefined".into()],
                    node.span,
                ),
                InfixOp::Div => return AnalyzedExpression::Int(left.wrapping_div(*right)),
                InfixOp::Rem if *right == 0 => self.error(
                    ErrorKind::Semantic,
                    format!("cannot calculate remainder of {left} with a divisor of 0"),
                    vec!["division by 0 is undefined".into()],
                    node.span,
                ),
                InfixOp::Rem => return AnalyzedExpression::Int(left.wrapping_rem(*right)),
                InfixOp::Pow => {
                    return AnalyzedExpression::Int(if *right < 0 {
                        0
                    } else {
                        left.wrapping_pow(*right as u32)
                    })
                }
                InfixOp::Eq => return AnalyzedExpression::Bool(left == right),
                InfixOp::Neq => return AnalyzedExpression::Bool(left != right),
                InfixOp::Lt => return AnalyzedExpression::Bool(left < right),
                InfixOp::Gt => return AnalyzedExpression::Bool(left > right),
                InfixOp::Lte => return AnalyzedExpression::Bool(left <= right),
                InfixOp::Gte => return AnalyzedExpression::Bool(left >= right),
                InfixOp::Shl | InfixOp::Shr => match *right {
                    0..=63 => {
                        return AnalyzedExpression::Int(match node.op == InfixOp::Shl {
                            true => left << right,
                            false => left >> right,
                        })
                    }
                    _ => self.error(
                        ErrorKind::Semantic,
                        format!("cannot shift by {right}"),
                        vec!["shifting by a number outside the range `0..=63` is undefined".into()],
                        node.span,
                    ),
                },
                InfixOp::BitOr => return AnalyzedExpression::Int(left | right),
                InfixOp::BitAnd => return AnalyzedExpression::Int(left & right),
                InfixOp::BitXor => return AnalyzedExpression::Int(left ^ right),
                _ => {}
            },
            (AnalyzedExpression::Float(left), AnalyzedExpression::Float(right)) => match node.op {
                InfixOp::Plus => return AnalyzedExpression::Float(left + right),
                InfixOp::Minus => return AnalyzedExpression::Float(left - right),
                InfixOp::Mul => return AnalyzedExpression::Float(left * right),
                InfixOp::Div => return AnalyzedExpression::Float(left / right),
                InfixOp::Eq => return AnalyzedExpression::Bool(left == right),
                InfixOp::Neq => return AnalyzedExpression::Bool(left != right),
                InfixOp::Lt => return AnalyzedExpression::Bool(left < right),
                InfixOp::Gt => return AnalyzedExpression::Bool(left > right),
                InfixOp::Lte => return AnalyzedExpression::Bool(left <= right),
                InfixOp::Gte => return AnalyzedExpression::Bool(left >= right),
                _ => {}
            },
            (AnalyzedExpression::Bool(left), AnalyzedExpression::Bool(right)) => match node.op {
                InfixOp::Eq => return AnalyzedExpression::Bool(left == right),
                InfixOp::Neq => return AnalyzedExpression::Bool(left != right),
                InfixOp::BitOr => return AnalyzedExpression::Bool(left | right),
                InfixOp::BitAnd => return AnalyzedExpression::Bool(left & right),
                InfixOp::BitXor => return AnalyzedExpression::Bool(left ^ right),
                InfixOp::And => return AnalyzedExpression::Bool(*left && *right),
                InfixOp::Or => return AnalyzedExpression::Bool(*left || *right),
                _ => {}
            },
            _ => {}
        }

        AnalyzedExpression::Infix(
            AnalyzedInfixExpr {
                result_type,
                lhs,
                op: node.op,
                rhs,
            }
            .into(),
        )
    }

    fn assign_type_error(&mut self, op: AssignOp, type_: Type, span: Span<'src>) -> Type {
        self.error(
            ErrorKind::Type,
            format!("assignment operator `{op}` does not allow values of type `{type_}`"),
            vec![],
            span,
        );
        Type::Unknown
    }

    fn assign_expr(&mut self, node: AssignExpr<'src>) -> AnalyzedExpression<'src> {
        let var_type = match self
            .scopes
            .iter_mut()
            .rev()
            .find_map(|scope| scope.get_mut(node.assignee.inner))
        {
            Some(var) => {
                var.mutated = true;
                let mut type_ = var.type_;
                if !var.mutable && node.assignee_ptr_count == 0 {
                    let span = var.span;
                    self.error(
                        ErrorKind::Semantic,
                        format!(
                            "cannot re-assign to immutable variable `{}`",
                            node.assignee.inner
                        ),
                        vec![],
                        node.span,
                    );
                    self.hint("variable not declared as `mut`", span);
                } else if node.assignee_ptr_count > 0 {
                    // using a pointer inside an assignment counts as use of the pointer variable
                    var.used = true;
                }

                for _ in 0..node.assignee_ptr_count {
                    type_ = match type_.sub_deref() {
                        Some(type_) => type_,
                        None => {
                            self.error(
                                ErrorKind::Type,
                                format!("cannot dereference a value of type `{type_}`"),
                                vec!["only pointers `*` can be dereferenced".into()],
                                node.assignee.span,
                            );
                            Type::Unknown
                        }
                    };
                }

                type_
            }
            None => match self.functions.get(node.assignee.inner) {
                Some(_) => {
                    self.error(
                        ErrorKind::Type,
                        "cannot assign to functions",
                        vec![],
                        node.assignee.span,
                    );
                    Type::Unknown
                }
                None if node.assignee.inner.is_empty() => Type::Unknown,
                None => {
                    self.error(
                        ErrorKind::Reference,
                        format!("use of undeclared name `{}`", node.assignee.inner),
                        vec![],
                        node.assignee.span,
                    );
                    Type::Unknown
                }
            },
        };

        let expr_span = node.expr.span();
        let expr = self.expression(node.expr);
        let result_type = match (node.op, var_type, expr.result_type()) {
            (_, Type::Unknown, _) | (_, _, Type::Unknown) => Type::Unknown,
            (_, _, Type::Never) => {
                self.warn_unreachable(node.span, expr_span, true);
                Type::Never
            }
            (_, left, right) if left != right => {
                self.error(
                    ErrorKind::Type,
                    format!("mismatched types: expected `{left}`, found `{right}`"),
                    vec![],
                    expr_span,
                );
                self.hint(
                    format!("this variable has type `{left}`"),
                    node.assignee.span,
                );
                Type::Unknown
            }
            (AssignOp::Plus | AssignOp::Minus, _, type_)
                if ![Type::Int(0), Type::Float(0), Type::Char(0)].contains(&type_) =>
            {
                self.assign_type_error(node.op, type_, expr_span)
            }
            (AssignOp::Mul | AssignOp::Div, _, type_)
                if ![Type::Int(0), Type::Float(0)].contains(&type_) =>
            {
                self.assign_type_error(node.op, type_, expr_span)
            }
            (AssignOp::Rem | AssignOp::Pow | AssignOp::Shl | AssignOp::Shr, _, type_)
                if type_ != Type::Int(0) =>
            {
                self.assign_type_error(node.op, type_, expr_span)
            }
            (AssignOp::BitOr | AssignOp::BitAnd | AssignOp::BitXor, _, type_)
                if ![Type::Int(0), Type::Bool(0)].contains(&type_) =>
            {
                self.assign_type_error(node.op, type_, expr_span)
            }
            (_, _, _) => Type::Unit,
        };

        AnalyzedExpression::Assign(
            AnalyzedAssignExpr {
                result_type,
                assignee: node.assignee.inner,
                assignee_ptr_count: node.assignee_ptr_count,
                op: node.op,
                expr,
            }
            .into(),
        )
    }

    fn call_expr(&mut self, node: CallExpr<'src>) -> AnalyzedExpression<'src> {
        let func = match (
            self.functions.get_mut(node.func.inner),
            self.builtin_functions.get(node.func.inner),
        ) {
            (Some(func), _) => {
                // only mark the function as used if it is called from outside of its body
                if self.curr_func_name != node.func.inner {
                    func.used = true;
                }
                Some((
                    func.return_type.inner.unwrap_or(Type::Unit),
                    func.params.clone(),
                ))
            }
            (_, Some(builtin)) => {
                self.used_builtins.insert(node.func.inner);
                let builtin = builtin.clone();
                let (result_type, args) = if node.args.len() != builtin.param_types.len() {
                    self.error(
                        ErrorKind::Reference,
                        format!(
                            "function `{}` takes {} arguments, however {} were supplied",
                            node.func.inner,
                            builtin.param_types.len(),
                            node.args.len()
                        ),
                        vec![],
                        node.span,
                    );
                    (builtin.return_type, vec![])
                } else {
                    let mut result_type = builtin.return_type;
                    let args = node
                        .args
                        .into_iter()
                        .zip(builtin.param_types)
                        .map(|(arg, param_type)| {
                            self.arg(arg, param_type, node.span, &mut result_type)
                        })
                        .collect();
                    (result_type, args)
                };

                return AnalyzedExpression::Call(
                    AnalyzedCallExpr {
                        result_type,
                        func: node.func.inner,
                        args,
                    }
                    .into(),
                );
            }
            (None, None) => {
                self.error(
                    ErrorKind::Reference,
                    format!("use of undeclared function `{}`", node.func.inner),
                    vec![format!(
                        "it can be declared like this: `fn {}(...) {{ ... }}`",
                        node.func.inner
                    )
                    .into()],
                    node.func.span,
                );
                None
            }
        };
        let (result_type, args) = match func {
            Some((func_type, func_params)) => {
                if node.args.len() != func_params.inner.len() {
                    self.error(
                        ErrorKind::Reference,
                        format!(
                            "function `{}` takes {} arguments, however {} were supplied",
                            node.func.inner,
                            func_params.inner.len(),
                            node.args.len()
                        ),
                        vec![],
                        node.span,
                    );
                    self.hint(
                        format!(
                            "function `{}` defined here with {} parameter{}",
                            node.func.inner,
                            func_params.inner.len(),
                            if func_params.inner.len() > 1 { "s" } else { "" }
                        ),
                        func_params.span,
                    );
                    (func_type, vec![])
                } else {
                    let mut result_type = func_type;
                    let args = node
                        .args
                        .into_iter()
                        .zip(func_params.inner)
                        .map(|(arg, param)| {
                            self.arg(arg, param.type_.inner, node.span, &mut result_type)
                        })
                        .collect();
                    (result_type, args)
                }
            }
            None => {
                let mut result_type = Type::Unknown;
                let args = node
                    .args
                    .into_iter()
                    .map(|arg| {
                        let arg = self.expression(arg);
                        if arg.result_type() == Type::Never {
                            result_type = Type::Never;
                        }
                        arg
                    })
                    .collect();
                (result_type, args)
            }
        };

        AnalyzedExpression::Call(
            AnalyzedCallExpr {
                result_type,
                func: node.func.inner,
                args,
            }
            .into(),
        )
    }

    fn arg(
        &mut self,
        arg: Expression<'src>,
        param_type: Type,
        call_span: Span<'src>,
        result_type: &mut Type,
    ) -> AnalyzedExpression<'src> {
        let arg_span = arg.span();
        let arg = self.expression(arg);

        match (arg.result_type(), param_type) {
            (Type::Unknown, _) | (_, Type::Unknown) => {}
            (Type::Never, _) => {
                self.warn_unreachable(call_span, arg_span, true);
                *result_type = Type::Never;
            }
            (arg_type, param_type) if arg_type != param_type => self.error(
                ErrorKind::Type,
                format!("mismatched types: expected `{param_type}`, found `{arg_type}`"),
                vec![],
                arg_span,
            ),
            _ => {}
        }

        arg
    }

    fn cast_expr(&mut self, node: CastExpr<'src>) -> AnalyzedExpression<'src> {
        let expr_span = node.expr.span();
        let expr = self.expression(node.expr);

        let result_type = match (expr.result_type(), node.type_.inner) {
            (Type::Unknown, _) => Type::Unknown,
            (Type::Never, _) => {
                self.warn_unreachable(node.span, expr_span, true);
                Type::Never
            }
            (left, right) if left == right => {
                self.info("unnecessary cast to same type", vec![], node.span);
                left
            }
            (
                Type::Int(0) | Type::Float(0) | Type::Bool(0) | Type::Char(0),
                Type::Int(0) | Type::Float(0) | Type::Bool(0) | Type::Char(0),
            ) => node.type_.inner,
            _ => {
                self.error(
                    ErrorKind::Type,
                    format!(
                        "invalid cast: cannot cast type `{}` to `{}",
                        expr.result_type(),
                        node.type_.inner
                    ),
                    vec![],
                    node.span,
                );
                Type::Unknown
            }
        };

        // evaluate constant expressions
        match (expr, result_type) {
            (AnalyzedExpression::Int(val), Type::Int(0)) => AnalyzedExpression::Int(val),
            (AnalyzedExpression::Int(val), Type::Float(0)) => AnalyzedExpression::Float(val as f64),
            (AnalyzedExpression::Int(val), Type::Bool(0)) => AnalyzedExpression::Bool(val != 0),
            (AnalyzedExpression::Int(val), Type::Char(0)) => {
                AnalyzedExpression::Char(val.clamp(0, 127) as u8)
            }
            (AnalyzedExpression::Float(val), Type::Int(0)) => AnalyzedExpression::Int(val as i64),
            (AnalyzedExpression::Float(val), Type::Float(0)) => AnalyzedExpression::Float(val),
            (AnalyzedExpression::Float(val), Type::Bool(0)) => AnalyzedExpression::Bool(val != 0.0),
            (AnalyzedExpression::Float(val), Type::Char(0)) => {
                AnalyzedExpression::Char(val.clamp(0.0, 127.0) as u8)
            }
            (AnalyzedExpression::Bool(val), Type::Int(0)) => AnalyzedExpression::Int(val as i64),
            (AnalyzedExpression::Bool(val), Type::Float(0)) => {
                AnalyzedExpression::Float(val as u8 as f64)
            }
            (AnalyzedExpression::Bool(val), Type::Bool(0)) => AnalyzedExpression::Bool(val),
            (AnalyzedExpression::Bool(val), Type::Char(0)) => AnalyzedExpression::Char(val as u8),
            (AnalyzedExpression::Char(val), Type::Int(0)) => AnalyzedExpression::Int(val as i64),
            (AnalyzedExpression::Char(val), Type::Float(0)) => {
                AnalyzedExpression::Float(val as f64)
            }
            (AnalyzedExpression::Char(val), Type::Bool(0)) => AnalyzedExpression::Bool(val != 0),
            (AnalyzedExpression::Char(val), Type::Char(0)) => AnalyzedExpression::Char(val),
            (expr, result_type) => AnalyzedExpression::Cast(
                AnalyzedCastExpr {
                    result_type,
                    expr,
                    type_: node.type_.inner,
                }
                .into(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rush_parser::{span, tree};

    fn program_test(
        parsed_tree: Program<'static>,
        analyzed_tree: AnalyzedProgram<'static>,
    ) -> Result<(), Vec<Diagnostic<'static>>> {
        let (tree, diagnostics) = dbg!(Analyzer::new("").analyze(parsed_tree))?;
        assert!(!diagnostics
            .iter()
            .any(|diag| matches!(diag.level, DiagnosticLevel::Error(_))));
        assert_eq!(tree, analyzed_tree);
        Ok(())
    }

    #[test]
    fn programs() -> Result<(), Vec<Diagnostic<'static>>> {
        // fn add(left: int, right: int) -> int { return left + right; } fn main() {}
        program_test(
            tree! {
                (Program @ 0..61,
                    functions: [
                        (FunctionDefinition @ 0..61,
                            name: ("add", @ 3..6),
                            params @ 6..29: [
                                (Parameter,
                                    mutable: false,
                                    name: ("left", @ 7..11),
                                    type: (Type::Int(0), @ 13..16)),
                                (Parameter,
                                    mutable: false,
                                    name: ("right", @ 18..23),
                                    type: (Type::Int(0), @ 25..28))],
                            return_type: (Some(Type::Int(0)), @ 33..36),
                            block: (Block @ 37..61,
                                stmts: [
                                    (ReturnStmt @ 39..59, (Some(InfixExpr @ 46..58,
                                        lhs: (Ident "left", @ 46..50),
                                        op: InfixOp::Plus,
                                        rhs: (Ident "right", @ 53..58))))],
                                expr: (None))),
                        (FunctionDefinition @ 62..74,
                            name: ("main", @ 65..69),
                            params @ 69..71: [],
                            return_type: (None, @ 70..73),
                            block: (Block @ 72..74,
                                stmts: [],
                                expr: (None)))],
                    globals: [])
            },
            analyzed_tree! {
                (Program,
                    globals: [],
                    functions: [
                        (FunctionDefinition,
                            used: false,
                            name: "add",
                            params: [
                                (Parameter,
                                    mutable: false,
                                    name: "left",
                                    type: Type::Int(0)  ),
                                (Parameter,
                                    mutable: false,
                                    name: "right",
                                    type: Type::Int(0)  )],
                            return_type: Type::Int(0),
                            block: (Block -> Type::Never,
                                stmts: [
                                    (ReturnStmt, (Some(InfixExpr -> Type::Int(0),
                                        lhs: (Ident -> Type::Int(0), "left"),
                                        op: InfixOp::Plus,
                                        rhs: (Ident -> Type::Int(0), "right"))))],
                                expr: (None)))],
                    main_fn: (Block -> Type::Unit,
                        stmts: [],
                        expr: (None)),
                    used_builtins: [])
            },
        )?;

        // fn main() { exit(1 + 2); }
        program_test(
            tree! {
                (Program @ 0..26,
                    functions: [
                        (FunctionDefinition @ 0..26,
                            name: ("main", @ 3..7),
                            params @ 7..9: [],
                            return_type: (None, @ 8..11),
                            block: (Block @ 10..26,
                                stmts: [
                                    (ExprStmt @ 12..24, (CallExpr @ 12..23,
                                        func: ("exit", @ 12..16),
                                        args: [
                                            (InfixExpr @ 17..22,
                                                lhs: (Int 1, @ 17..18),
                                                op: InfixOp::Plus,
                                                rhs: (Int 2, @ 21..22))]))],
                                expr: (None)))],
                    globals: [])
            },
            analyzed_tree! {
                (Program,
                    globals: [],
                    functions: [],
                    main_fn: (Block -> Type::Never,
                        stmts: [
                            (ExprStmt, (CallExpr -> Type::Never,
                                func: "exit",
                                args: [(Int 3)]))],
                        expr: (None)),
                    used_builtins: ["exit"])
            },
        )?;

        Ok(())
    }
}

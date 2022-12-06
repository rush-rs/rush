use std::{collections::HashMap, mem};

use rush_analyzer::{ast::*, AssignOp, InfixOp, PrefixOp, Type};

use crate::{
    instructions, types,
    utils::{self, Leb128},
};

#[derive(Debug, Default)]
pub struct Compiler<'src> {
    /// The instructions of the currently compiling function
    pub(crate) function_body: Vec<u8>,
    /// The locals of the currently compiling function
    pub(crate) locals: Vec<Vec<u8>>,
    /// The count of parameters the current function takes
    pub(crate) param_count: usize,
    /// Function bodies to append to the code section after the user defined functions
    pub(crate) builtins_code: Vec<Vec<u8>>,
    /// The count of currently nested blocks since the last loop
    pub(crate) block_count: usize,
    /// The labels for `break` statements in the current loop. For `loop` and `while` loops
    /// this is `1` and for `for` loops it is `2`.
    pub(crate) break_labels: Vec<usize>,

    /// Maps variable names to `Option<local_idx>`, or `None` when of type `()`
    pub(crate) scopes: Vec<HashMap<&'src str, Option<Vec<u8>>>>,
    /// Maps global variable names to `global_idx`
    pub(crate) global_scope: HashMap<&'src str, Vec<u8>>,
    /// Maps function names to their index encoded as unsigned LEB128
    pub(crate) functions: HashMap<&'src str, Vec<u8>>,
    /// Maps builtin function names to their index encoded as unsigned LEB128
    pub(crate) builtin_functions: HashMap<&'static str, Vec<u8>>,
    /// The number of imports this module uses
    pub(crate) import_count: usize,

    pub(crate) type_section: Vec<Vec<u8>>,     // 1
    pub(crate) import_section: Vec<Vec<u8>>,   // 2
    pub(crate) function_section: Vec<Vec<u8>>, // 3
    pub(crate) table_section: Vec<Vec<u8>>,    // 4
    pub(crate) memory_section: Vec<Vec<u8>>,   // 5
    pub(crate) global_section: Vec<Vec<u8>>,   // 6
    pub(crate) export_section: Vec<Vec<u8>>,   // 7
    pub(crate) start_section: Vec<u8>,         // 8
    pub(crate) element_section: Vec<Vec<u8>>,  // 9
    pub(crate) code_section: Vec<Vec<u8>>,     // 10
    pub(crate) data_section: Vec<Vec<u8>>,     // 11
    pub(crate) data_count_section: Vec<u8>,    // 12

    pub(crate) function_names: Vec<Vec<u8>>,
    pub(crate) imported_function_names: Vec<Vec<u8>>,
    /// List of `(func_idx, list_of_locals)`
    pub(crate) local_names: Vec<(Vec<u8>, Vec<Vec<u8>>)>,
    /// The index of the currently compiling function in the `local_names` vec
    pub(crate) curr_func_idx: usize,
    pub(crate) global_names: Vec<Vec<u8>>,
}

enum Variable<'compiler> {
    Global(&'compiler Vec<u8>),
    Local(&'compiler Option<Vec<u8>>),
}

impl<'src> Compiler<'src> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compile(mut self, tree: AnalyzedProgram<'src>) -> Vec<u8> {
        // set count of imports needed
        self.import_count = tree.used_builtins.len();
        // add required imports
        for import in &tree.used_builtins {
            if *import == "exit" {
                self.__wasi_exit(false);
            }
        }

        // compile program
        self.program(tree);

        // add blank memory
        self.memory_section.push(vec![0, 0]);

        // export memory
        self.export_section.push(
            [
                &[6][..],  // string len
                b"memory", // export name
                &[
                    2, // export kind (2 = memory)
                    0, // index in memory section
                ],
            ]
            .concat(),
        );

        // concat function name vectors
        self.imported_function_names
            .append(&mut self.function_names);

        // concat sections
        [
            &b"\0asm"[..],        // magic
            &1_i32.to_le_bytes(), // spec version 1
            &Self::section(1, self.type_section),
            &Self::section(2, self.import_section),
            &Self::section(3, self.function_section),
            &Self::section(4, self.table_section),
            &Self::section(5, self.memory_section),
            &Self::section(6, self.global_section),
            &Self::section(7, self.export_section),
            &self.start_section,
            &Self::section(9, self.element_section),
            &Self::section(10, self.code_section),
            &Self::section(11, self.data_section),
            &self.data_count_section,
            &Self::name_section(
                self.imported_function_names,
                self.local_names,
                self.global_names,
            ),
        ]
        .concat()
    }

    fn section(id: u8, section: Vec<Vec<u8>>) -> Vec<u8> {
        if section.is_empty() {
            return vec![];
        }

        // start new buf with section id
        let mut buf = vec![id];

        // add vector
        buf.append(&mut Self::vector(section, true));

        // return
        buf
    }

    fn vector(vec: Vec<Vec<u8>>, add_byte_count: bool) -> Vec<u8> {
        let mut buf = vec![];

        // combine vector contents
        let mut combined_bytes = vec.concat();

        // get length of vector
        let mut vector_len = vec.len().to_uleb128();

        // add byte count
        if add_byte_count {
            buf.append(&mut (combined_bytes.len() + vector_len.len()).to_uleb128());
        }

        // add vector length
        buf.append(&mut vector_len);

        // add contents
        buf.append(&mut combined_bytes);

        // return
        buf
    }

    fn name_section(
        function_names: Vec<Vec<u8>>,
        local_names: Vec<(Vec<u8>, Vec<Vec<u8>>)>,
        global_names: Vec<Vec<u8>>,
    ) -> Vec<u8> {
        let contents = [
            &[4][..],                          // string len
            b"name",                           // custom section name "name"
            &Self::section(1, function_names), // function names subsection
            // local names subsection
            &Self::section(
                2,
                local_names
                    .into_iter()
                    .map(|(func_idx, locals)| [func_idx, Self::vector(locals, false)].concat())
                    .collect(),
            ),
            &Self::section(7, global_names), // global names subsection
        ]
        .concat();
        [
            &[0][..],                     // section id
            &contents.len().to_uleb128(), // section size
            &contents,                    // section contents
        ]
        .concat()
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) -> Option<HashMap<&'src str, Option<Vec<u8>>>> {
        self.scopes.pop()
    }

    fn curr_scope(&mut self) -> &mut HashMap<&'src str, Option<Vec<u8>>> {
        self.scopes.last_mut().unwrap()
    }

    fn find_var(&self, name: &'src str) -> Variable<'_> {
        for scope in self.scopes.iter().rev() {
            if let Some(idx) = scope.get(name) {
                return Variable::Local(idx);
            }
        }
        if let Some(idx) = self.global_scope.get(name) {
            return Variable::Global(idx);
        }
        unreachable!("the analyzer guarantees valid variable references");
    }

    /////////////////////////

    fn program(&mut self, node: AnalyzedProgram<'src>) {
        // add globals
        for global in &node.globals {
            // get index
            let global_idx = self.global_scope.len().to_uleb128();

            // add name to name section
            self.global_names.push(
                [
                    &global_idx[..],                 // function index
                    &global.name.len().to_uleb128(), // string len
                    global.name.as_bytes(),          // name
                ]
                .concat(),
            );

            // save index in global scope
            self.global_scope.insert(global.name, global_idx);

            let mut buf = vec![
                utils::type_to_byte(global.expr.result_type())
                    .expect("globals cannot be `()` or `!`"), // type of global
                global.mutable as u8,
            ];
            match global.expr {
                AnalyzedExpression::Int(num) => {
                    buf.push(instructions::I64_CONST);
                    num.write_sleb128(&mut buf);
                }
                AnalyzedExpression::Float(num) => {
                    buf.push(instructions::F64_CONST);
                    buf.extend_from_slice(&num.to_le_bytes());
                }
                AnalyzedExpression::Bool(bool) => {
                    buf.push(instructions::I32_CONST);
                    bool.write_sleb128(&mut buf);
                }
                AnalyzedExpression::Char(num) => {
                    buf.push(instructions::I32_CONST);
                    num.write_sleb128(&mut buf);
                }
                _ => panic!("globals cannot be `()` or `!`"),
            }
            buf.push(instructions::END);
            self.global_section.push(buf);
        }

        // add main fn signature
        {
            // add to self.functions map
            let func_idx = self.import_count.to_uleb128();
            self.functions.insert("main", func_idx.clone());

            // add signature to type section
            self.type_section.push(vec![
                types::FUNC,
                0, // num of params
                0, // num of return vals
            ]);

            // index of signature in type section
            self.function_section.push(self.import_count.to_uleb128());

            // add name to name section
            self.function_names.push(
                [
                    &func_idx[..], // function index
                    &[4],          // string len
                    b"main",       // name
                ]
                .concat(),
            );

            // no params in name section
            self.local_names.push((func_idx, vec![]));
        }

        // add other function signatures
        for func in &node.functions {
            self.function_signature(func);
        }

        // compile functions
        self.main_fn(node.main_fn);
        for (idx, func) in node
            .functions
            .into_iter()
            .filter(|func| func.used)
            .enumerate()
        {
            self.curr_func_idx = idx + 1;
            self.function_definition(func);
        }

        // push bodies of builtin functions
        self.code_section.append(&mut self.builtins_code);
    }

    fn main_fn(&mut self, body: AnalyzedBlock<'src>) {
        let main_idx = self.import_count.to_uleb128();

        // export main func as WASI `_start` func
        self.export_section.push(
            [
                &[6][..],  // string len
                b"_start", // name of export
                &[0],      // export kind (0 = func)
                &main_idx, // index of func
            ]
            .concat(),
        );

        // point to main func in start section
        self.start_section = [
            &[8][..],                     // section id
            &main_idx.len().to_uleb128(), // section len
            &main_idx,                    // index of main func
        ]
        .concat();

        // set param_count to 0
        self.param_count = 0;

        // function body
        self.push_scope();
        self.function_body(body);
        self.pop_scope();
    }

    fn function_signature(&mut self, node: &AnalyzedFunctionDefinition<'src>) {
        // skip unused functions
        if !node.used {
            return;
        }

        // add to self.functions map
        let func_idx = (self.function_section.len() + self.import_count).to_uleb128();
        self.functions.insert(node.name, func_idx.clone());

        // start new buf vor func signature
        let mut buf = vec![types::FUNC];

        // add param types
        let mut param_names = vec![];
        let mut params: Vec<_> = node
            .params
            .iter()
            .filter_map(|param| {
                let wasm_type = utils::type_to_byte(param.type_);
                if wasm_type.is_some() {
                    param_names.push(
                        [
                            &param_names.len().to_uleb128()[..], // local index
                            &param.name.len().to_uleb128(),      // string len
                            param.name.as_bytes(),               // param name
                        ]
                        .concat(),
                    );
                }
                wasm_type
            })
            .collect();
        params.len().write_uleb128(&mut buf);
        buf.append(&mut params);

        // add return type
        match utils::type_to_byte(node.return_type) {
            // unit type => no return values
            None => buf.push(0),
            // any other type => 1 return value of that type
            Some(return_type) => {
                buf.push(1); // 1 return value
                buf.push(return_type); // the return type
            }
        }

        // push to type section and save idx
        let func_type_idx = self.type_section.len();
        self.type_section.push(buf);

        // index of type in type_section
        self.function_section.push(func_type_idx.to_uleb128());

        // add name to name section
        self.function_names.push(
            [
                &func_idx[..],                 // function index
                &node.name.len().to_uleb128(), // string len
                node.name.as_bytes(),          // name
            ]
            .concat(),
        );

        // add param names to name section
        self.local_names.push((func_idx, param_names));
    }

    fn function_definition(&mut self, node: AnalyzedFunctionDefinition<'src>) {
        // reset param count
        self.param_count = 0;

        // add params to new scope
        let mut scope = HashMap::new();
        for param in node.params.into_iter() {
            scope.insert(
                param.name,
                utils::type_to_byte(param.type_).map(|_| {
                    let res = self.param_count.to_uleb128();
                    self.param_count += 1;
                    res
                }),
            );
        }

        // function body
        self.scopes.push(scope);
        self.function_body(node.block);
        self.pop_scope();
    }

    fn function_body(&mut self, node: AnalyzedBlock<'src>) {
        for stmt in node.stmts {
            self.statement(stmt);
        }
        if let Some(expr) = node.expr {
            self.expression(expr);
        }
        self.function_body.push(instructions::END);

        // take self.locals
        let locals = Self::vector(mem::take(&mut self.locals), false);

        // push function
        self.code_section.push(
            [
                // body size
                (self.function_body.len() + locals.len()).to_uleb128(),
                // locals
                locals,
                // function body
                mem::take(&mut self.function_body),
            ]
            .concat(),
        );
    }

    /////////////////////////

    fn statement(&mut self, node: AnalyzedStatement<'src>) {
        match node {
            AnalyzedStatement::Let(node) => self.let_stmt(node),
            AnalyzedStatement::Return(node) => self.return_stmt(node),
            AnalyzedStatement::Loop(node) => self.loop_stmt(node),
            AnalyzedStatement::While(node) => self.while_stmt(node),
            AnalyzedStatement::For(node) => self.for_stmt(node),
            AnalyzedStatement::Break => {
                self.function_body.push(instructions::BR); // jump
                (self.block_count
                    + self
                        .break_labels
                        .last()
                        .expect("the analyzer guarantees loops around break-statements"))
                .write_uleb128(&mut self.function_body); // to end of block around loop
            }
            AnalyzedStatement::Continue => {
                self.function_body.push(instructions::BR); // jump
                self.block_count.write_uleb128(&mut self.function_body); // to start of loop
            }
            AnalyzedStatement::Expr(expr) => {
                let expr_type = expr.result_type();
                self.expression(expr);
                if !matches!(expr_type, Type::Unit | Type::Never) {
                    self.function_body.push(instructions::DROP);
                }
            }
        }
    }

    fn let_stmt(&mut self, node: AnalyzedLetStmt<'src>) {
        let wasm_type = utils::type_to_byte(node.expr.result_type());
        let local_idx = (self.locals.len() + self.param_count).to_uleb128();
        if let Some(byte) = wasm_type {
            self.locals.push(vec![1, byte]);
        }
        self.curr_scope()
            .insert(node.name, wasm_type.map(|_| local_idx.clone()));

        // add variable name to name section
        if wasm_type.is_some() {
            self.local_names[self.curr_func_idx].1.push(
                [
                    &local_idx[..],                // local index
                    &node.name.len().to_uleb128(), // string len
                    node.name.as_bytes(),
                ]
                .concat(),
            );
        }

        self.expression(node.expr);
        if wasm_type.is_some() {
            self.function_body.push(instructions::LOCAL_SET);
            self.function_body.extend_from_slice(&local_idx);
        }
    }

    fn return_stmt(&mut self, node: AnalyzedReturnStmt<'src>) {
        if let Some(expr) = node {
            self.expression(expr);
        }
        self.function_body.push(instructions::RETURN);
    }

    fn loop_stmt(&mut self, node: AnalyzedLoopStmt<'src>) {
        // store current block count and set break label
        let prev_block_count = self.block_count;
        self.block_count = 0;
        self.break_labels.push(1);

        self.function_body.push(instructions::BLOCK); // outer block to jump to with `break`
        self.function_body.push(types::VOID); // with result `()`
        self.function_body.push(instructions::LOOP); // loop to jump to with `continue`
        self.function_body.push(types::VOID); // with result `()`

        self.block_expr(node.block, true);
        self.function_body.push(instructions::BR); // jump
        self.function_body.push(0); // to start of loop

        self.function_body.push(instructions::END); // end of loop
        self.function_body.push(instructions::END); // end of block

        // restore block count and loop labels
        self.block_count = prev_block_count;
        self.break_labels.pop();
    }

    fn while_stmt(&mut self, node: AnalyzedWhileStmt<'src>) {
        // store current block count and set break label
        let prev_block_count = self.block_count;
        self.block_count = 0;
        self.break_labels.push(1);

        self.function_body.push(instructions::BLOCK); // outer block to jump to with `break`
        self.function_body.push(types::VOID); // with result `()`
        self.function_body.push(instructions::LOOP); // loop to jump to with `continue`
        self.function_body.push(types::VOID); // with result `()`

        self.expression(node.cond); // compile condition
        self.function_body.push(instructions::I32_EQZ); // negate result
        self.function_body.push(instructions::BR_IF); // jump if cond is not true
        self.function_body.push(1); // to end of outer block

        self.block_expr(node.block, true);
        self.function_body.push(instructions::BR); // jump
        self.function_body.push(0); // to start of loop

        self.function_body.push(instructions::END); // end of loop
        self.function_body.push(instructions::END); // end of block

        // restore block count and loop labels
        self.block_count = prev_block_count;
        self.break_labels.pop();
    }

    fn for_stmt(&mut self, node: AnalyzedForStmt<'src>) {
        // store current block count and set break label
        let prev_block_count = self.block_count;
        self.block_count = 0;
        self.break_labels.push(2);

        // compile the initializer
        let wasm_type = utils::type_to_byte(node.initializer.result_type());
        let local_idx = (self.locals.len() + self.param_count).to_uleb128();
        if let Some(byte) = wasm_type {
            self.locals.push(vec![1, byte]);
        }

        // add init variable name to name section
        if wasm_type.is_some() {
            self.local_names[self.curr_func_idx].1.push(
                [
                    &local_idx[..],                 // local index
                    &node.ident.len().to_uleb128(), // string len
                    node.ident.as_bytes(),
                ]
                .concat(),
            );
        }

        self.expression(node.initializer);
        self.function_body.push(instructions::LOCAL_SET);
        self.function_body.extend_from_slice(&local_idx);

        // create new scope with induction variable
        self.scopes.push(HashMap::from([(
            node.ident,
            wasm_type.map(|_| local_idx.clone()),
        )]));

        self.function_body.push(instructions::BLOCK); // outer block to jump to with `break`
        self.function_body.push(types::VOID); // with result `()`
        self.function_body.push(instructions::LOOP); // loop to jump to at end of iteration
        self.function_body.push(types::VOID); // with result `()`
        self.function_body.push(instructions::BLOCK); // block to jump to with continue
        self.function_body.push(types::VOID); // with result `()`

        self.expression(node.cond); // compile condition
        self.function_body.push(instructions::I32_EQZ); // negate result
        self.function_body.push(instructions::BR_IF); // jump if cond is not true
        self.function_body.push(2); // to end of outer block

        self.block_expr(node.block, false); // the loop body

        self.function_body.push(instructions::END); // end of block

        self.expression(node.update); // loop update expression

        self.function_body.push(instructions::BR); // jump
        self.function_body.push(0); // to start of loop

        self.function_body.push(instructions::END); // end of loop
        self.function_body.push(instructions::END); // end of block

        // restore block count and loop labels
        self.block_count = prev_block_count;
        self.break_labels.pop();

        // pop scope
        self.pop_scope();
    }

    fn expression(&mut self, node: AnalyzedExpression<'src>) {
        let diverges = node.result_type() == Type::Never;
        match node {
            AnalyzedExpression::Block(node) => self.block_expr(*node, true),
            AnalyzedExpression::If(node) => self.if_expr(*node),
            AnalyzedExpression::Int(value) => {
                // `int`s are stored as signed `i64`
                self.function_body.push(instructions::I64_CONST);
                value.write_sleb128(&mut self.function_body);
            }
            AnalyzedExpression::Float(value) => {
                // `float`s are stored as `f64`
                self.function_body.push(instructions::F64_CONST);
                self.function_body.extend(value.to_le_bytes());
            }
            AnalyzedExpression::Bool(value) => {
                // `bool`s are stored as unsigned `i32`
                self.function_body.push(instructions::I32_CONST);
                value.write_sleb128(&mut self.function_body);
            }
            AnalyzedExpression::Char(value) => {
                // `char`s are stored as unsigned `i32`
                self.function_body.push(instructions::I32_CONST);
                value.write_sleb128(&mut self.function_body);
            }
            AnalyzedExpression::Ident(ident) => match self.find_var(ident.ident) {
                Variable::Global(global_idx) => {
                    let global_idx = global_idx.clone();
                    self.function_body.push(instructions::GLOBAL_GET);
                    self.function_body.extend_from_slice(&global_idx);
                }
                Variable::Local(Some(local_idx)) => {
                    let local_idx = local_idx.clone();
                    self.function_body.push(instructions::LOCAL_GET);
                    self.function_body.extend_from_slice(&local_idx);
                }
                // unit type requires no instructions
                Variable::Local(None) => {}
            },
            AnalyzedExpression::Prefix(node) => self.prefix_expr(*node),
            AnalyzedExpression::Infix(node) => self.infix_expr(*node),
            AnalyzedExpression::Assign(node) => self.assign_expr(*node),
            AnalyzedExpression::Call(node) => self.call_expr(*node),
            AnalyzedExpression::Cast(node) => self.cast_expr(*node),
            AnalyzedExpression::Grouped(expr) => self.expression(*expr),
        }
        if diverges {
            self.function_body.push(instructions::UNREACHABLE);
        }
    }

    fn block_expr(&mut self, node: AnalyzedBlock<'src>, new_scope: bool) {
        if new_scope {
            self.push_scope();
        }
        for stmt in node.stmts {
            self.statement(stmt);
        }
        if let Some(expr) = node.expr {
            self.expression(expr);
        }
        if new_scope {
            self.pop_scope();
        }
    }

    fn if_expr(&mut self, node: AnalyzedIfExpr<'src>) {
        // compile condition
        self.expression(node.cond);

        self.function_body.push(instructions::IF);
        match utils::type_to_byte(node.result_type) {
            Some(byte) => self.function_body.push(byte),
            None => self.function_body.push(types::VOID),
        }

        self.block_count += 1;
        self.block_expr(node.then_block, true);

        if let Some(else_block) = node.else_block {
            self.function_body.push(instructions::ELSE);
            self.block_expr(else_block, true);
        }

        self.block_count -= 1;
        self.function_body.push(instructions::END);
    }

    fn prefix_expr(&mut self, node: AnalyzedPrefixExpr<'src>) {
        // match op and expr type
        match (node.op, node.expr.result_type()) {
            (_, Type::Never) => self.expression(node.expr),
            (PrefixOp::Not, Type::Bool) => {
                // compile expression
                self.expression(node.expr);

                // negate
                self.function_body.push(instructions::I32_EQZ);
            }
            (PrefixOp::Neg, Type::Int) => {
                // push constant 0
                self.function_body.push(instructions::I64_CONST);
                self.function_body.push(0);

                // compile expression
                self.expression(node.expr);

                // push subtract
                self.function_body.push(instructions::I64_SUB);
            }
            (PrefixOp::Neg, Type::Float) => {
                // compile expression
                self.expression(node.expr);

                self.function_body.push(instructions::F64_NEG);
            }
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }

    fn infix_expr(&mut self, node: AnalyzedInfixExpr<'src>) {
        match node.op {
            InfixOp::And => {
                self.expression(node.lhs);
                self.function_body.extend([
                    // if lhs is not true
                    instructions::I32_EQZ,
                    instructions::IF,
                    types::I32,
                    // then return true
                    instructions::I32_CONST,
                    0,
                    // else return rhs
                    instructions::ELSE,
                ]);
                self.expression(node.rhs);
                // end if
                self.function_body.push(instructions::END);

                return;
            }
            InfixOp::Or => {
                self.expression(node.lhs);
                self.function_body.extend([
                    // if lhs is true
                    instructions::IF,
                    types::I32,
                    // then return true
                    instructions::I32_CONST,
                    1,
                    // else return rhs
                    instructions::ELSE,
                ]);
                self.expression(node.rhs);
                // end if
                self.function_body.push(instructions::END);

                return;
            }
            _ => {}
        }

        // save type of exprs
        let lhs_type = node.lhs.result_type();
        let rhs_type = node.rhs.result_type();

        // compile left and right expression
        self.expression(node.lhs);
        self.expression(node.rhs);

        // match on op and type (analyzer guarantees same type for lhs and rhs)
        let instruction = match (node.op, lhs_type) {
            _ if lhs_type == Type::Never || rhs_type == Type::Never => return,
            (InfixOp::Plus, Type::Int) => instructions::I64_ADD,
            (InfixOp::Plus, Type::Float) => instructions::F64_ADD,
            (InfixOp::Plus, Type::Char) => {
                self.function_body.push(instructions::I32_ADD);
                self.function_body.push(instructions::I32_CONST);
                0x7f.write_sleb128(&mut self.function_body);
                self.function_body.push(instructions::I32_AND);
                return;
            }
            (InfixOp::Minus, Type::Int) => instructions::I64_SUB,
            (InfixOp::Minus, Type::Float) => instructions::F64_SUB,
            (InfixOp::Minus, Type::Char) => {
                self.function_body.push(instructions::I32_SUB);
                self.function_body.push(instructions::I32_CONST);
                0x7f.write_sleb128(&mut self.function_body);
                self.function_body.push(instructions::I32_AND);
                return;
            }
            (InfixOp::Mul, Type::Int) => instructions::I64_MUL,
            (InfixOp::Mul, Type::Float) => instructions::F64_MUL,
            (InfixOp::Div, Type::Int) => instructions::I64_DIV_S,
            (InfixOp::Div, Type::Float) => instructions::F64_DIV,
            (InfixOp::Rem, Type::Int) => instructions::I64_REM_S,
            (InfixOp::Pow, Type::Int) => return self.__rush_internal_pow_int(),
            (InfixOp::Eq, Type::Int) => instructions::I64_EQ,
            (InfixOp::Eq, Type::Float) => instructions::F64_EQ,
            (InfixOp::Eq, Type::Bool) => instructions::I32_EQ,
            (InfixOp::Eq, Type::Char) => instructions::I32_EQ,
            (InfixOp::Neq, Type::Int) => instructions::I64_NE,
            (InfixOp::Neq, Type::Float) => instructions::F64_NE,
            (InfixOp::Neq, Type::Bool) => instructions::I32_NE,
            (InfixOp::Neq, Type::Char) => instructions::I32_NE,
            (InfixOp::Lt, Type::Int) => instructions::I64_LT_S,
            (InfixOp::Lt, Type::Float) => instructions::F64_LT,
            (InfixOp::Gt, Type::Int) => instructions::I64_GT_S,
            (InfixOp::Gt, Type::Float) => instructions::F64_GT,
            (InfixOp::Lte, Type::Int) => instructions::I64_LE_S,
            (InfixOp::Lte, Type::Float) => instructions::F64_LE,
            (InfixOp::Gte, Type::Int) => instructions::I64_GE_S,
            (InfixOp::Gte, Type::Float) => instructions::F64_GE,
            (InfixOp::Shl, Type::Int) => instructions::I64_SHL,
            (InfixOp::Shr, Type::Int) => instructions::I64_SHR_S,
            (InfixOp::BitOr, Type::Int) => instructions::I64_OR,
            (InfixOp::BitOr, Type::Bool) => instructions::I32_OR,
            (InfixOp::BitAnd, Type::Int) => instructions::I64_AND,
            (InfixOp::BitAnd, Type::Bool) => instructions::I32_AND,
            (InfixOp::BitXor, Type::Int) => instructions::I64_XOR,
            (InfixOp::BitXor, Type::Bool) => instructions::I32_XOR,
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        };
        self.function_body.push(instruction);
    }

    fn assign_expr(&mut self, node: AnalyzedAssignExpr<'src>) {
        let (get_instr, set_instr, idx) = match self.find_var(node.assignee) {
            Variable::Global(idx) => (
                instructions::GLOBAL_GET,
                instructions::GLOBAL_SET,
                idx.clone(),
            ),
            Variable::Local(Some(idx)) => (
                instructions::LOCAL_GET,
                instructions::LOCAL_SET,
                idx.clone(),
            ),
            Variable::Local(None) => {
                self.expression(node.expr);
                return;
            }
        };

        // get the current value if assignment is more than just `=`
        if node.op != AssignOp::Basic {
            self.function_body.push(get_instr);
            self.function_body.extend_from_slice(&idx);
        }

        // save expr_type
        let expr_type = node.expr.result_type();

        // compile rhs
        self.expression(node.expr);

        // calculate new value for non-basic assignments
        'op: {
            if node.op != AssignOp::Basic {
                // match on op and type (analyzer guarantees same type for variable and expr)
                let instruction = match (node.op, expr_type) {
                    (_, Type::Never) => break 'op,
                    (AssignOp::Plus, Type::Int) => instructions::I64_ADD,
                    (AssignOp::Plus, Type::Float) => instructions::F64_ADD,
                    (AssignOp::Plus, Type::Char) => {
                        self.function_body.push(instructions::I32_ADD);
                        self.function_body.push(instructions::I32_CONST);
                        0x7f.write_sleb128(&mut self.function_body);
                        self.function_body.push(instructions::I32_AND);
                        break 'op;
                    }
                    (AssignOp::Minus, Type::Int) => instructions::I64_SUB,
                    (AssignOp::Minus, Type::Float) => instructions::F64_SUB,
                    (AssignOp::Minus, Type::Char) => {
                        self.function_body.push(instructions::I32_SUB);
                        self.function_body.push(instructions::I32_CONST);
                        0x7f.write_sleb128(&mut self.function_body);
                        self.function_body.push(instructions::I32_AND);
                        break 'op;
                    }
                    (AssignOp::Mul, Type::Int) => instructions::I64_MUL,
                    (AssignOp::Mul, Type::Float) => instructions::F64_MUL,
                    (AssignOp::Div, Type::Int) => instructions::I64_DIV_S,
                    (AssignOp::Div, Type::Float) => instructions::F64_DIV,
                    (AssignOp::Rem, Type::Int) => instructions::I64_REM_S,
                    (AssignOp::Pow, Type::Int) => {
                        self.__rush_internal_pow_int();
                        break 'op;
                    }
                    (AssignOp::Shl, Type::Int) => instructions::I64_SHL,
                    (AssignOp::Shr, Type::Int) => instructions::I64_SHR_S,
                    (AssignOp::BitOr, Type::Int) => instructions::I64_OR,
                    (AssignOp::BitOr, Type::Bool) => instructions::I32_OR,
                    (AssignOp::BitAnd, Type::Int) => instructions::I64_AND,
                    (AssignOp::BitAnd, Type::Bool) => instructions::I32_AND,
                    (AssignOp::BitXor, Type::Int) => instructions::I64_XOR,
                    (AssignOp::BitXor, Type::Bool) => instructions::I32_XOR,
                    _ => unreachable!("the analyzer guarantees one of the above to match"),
                };
                self.function_body.push(instruction);
            }
        }

        // set local to new value
        self.function_body.push(set_instr);
        self.function_body.extend_from_slice(&idx);
    }

    fn call_expr(&mut self, node: AnalyzedCallExpr<'src>) {
        for arg in node.args {
            self.expression(arg);
        }

        match self.functions.get(node.func) {
            Some(func) => {
                self.function_body.push(instructions::CALL);
                self.function_body.extend_from_slice(func);
            }
            None => match node.func {
                "exit" => self.__wasi_exit(true),
                _ => unreachable!("the analyzer guarantees one of the above to match"),
            },
        }
    }

    fn cast_expr(&mut self, node: AnalyzedCastExpr<'src>) {
        // save expr type
        let expr_type = node.expr.result_type();

        // compile expression
        self.expression(node.expr);

        // match source and dest types
        match (expr_type, node.type_) {
            // expression diverges at runtime: do nothing
            (Type::Never, _) => {}
            // type does not change: do nothing
            (source, dest) if source == dest => {}
            (Type::Bool, Type::Char) => {}

            (Type::Int, Type::Float) => self.function_body.push(instructions::F64_CONVERT_I64_S),
            (Type::Int, Type::Bool) => {
                // push constant 0
                self.function_body.push(instructions::I64_CONST);
                self.function_body.push(0);

                // true if != 0
                self.function_body.push(instructions::I64_NE);
            }
            (Type::Int, Type::Char) => self.__rush_internal_cast_int_to_char(),
            (Type::Float, Type::Int) => {
                self.function_body.extend(instructions::I64_TRUNC_SAT_F64_S);
            }
            (Type::Float, Type::Bool) => {
                // push constant 0
                self.function_body.push(instructions::F64_CONST);
                self.function_body.extend(0_f64.to_le_bytes());

                // true if != 0
                self.function_body.push(instructions::F64_NE);
            }
            (Type::Float, Type::Char) => self.__rush_internal_cast_float_to_char(),
            (Type::Bool, Type::Int) => self.function_body.push(instructions::I64_EXTEND_I32_U),
            (Type::Bool, Type::Float) => self.function_body.push(instructions::F64_CONVERT_I32_U),
            (Type::Char, Type::Int) => self.function_body.push(instructions::I64_EXTEND_I32_U),
            (Type::Char, Type::Float) => self.function_body.push(instructions::F64_CONVERT_I32_U),
            (Type::Char, Type::Bool) => {
                // push constant 0
                self.function_body.push(instructions::I32_CONST);
                self.function_body.push(0);

                // true if != 0
                self.function_body.push(instructions::I32_NE);
            }
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

use std::{collections::HashMap, mem};

use rush_analyzer::{ast::*, AssignOp, InfixOp, Type};

use crate::{
    instructions, types,
    utils::{self, Leb128},
};

#[derive(Debug, Default)]
pub struct Compiler<'src> {
    function_body: Vec<u8>,
    locals: Vec<Vec<u8>>,

    /// Maps variable names to `Option<(type, local_idx)>`, or `None` when of type `()`
    // TODO: maybe remove type
    scope: HashMap<&'src str, Option<(u8, Vec<u8>)>>,
    /// Maps function names to their index encoded as unsigned LEB128
    functions: HashMap<&'src str, Vec<u8>>,
    /// The count of parameters the current function takes
    param_count: usize,

    type_section: Vec<Vec<u8>>,     // 1
    import_section: Vec<Vec<u8>>,   // 2
    function_section: Vec<Vec<u8>>, // 3
    table_section: Vec<Vec<u8>>,    // 4
    memory_section: Vec<Vec<u8>>,   // 5
    global_section: Vec<Vec<u8>>,   // 6
    export_section: Vec<Vec<u8>>,   // 7
    start_section: Vec<u8>,         // 8
    element_section: Vec<Vec<u8>>,  // 9
    code_section: Vec<Vec<u8>>,     // 10
    data_section: Vec<Vec<u8>>,     // 11
    data_count_section: Vec<u8>,    // 12
}

impl<'src> Compiler<'src> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compile(mut self, tree: AnalyzedProgram<'src>) -> Vec<u8> {
        self.program(tree);
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
            // TODO: start and data_count section
            // &Self::section(8, self.start_section),
            &Self::section(9, self.element_section),
            &Self::section(10, self.code_section),
            &Self::section(11, self.data_section),
            // &Self::section(12, self.data_count_section),
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

    // TODO: maybe remove `add_byte_count` param
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

    /////////////////////////

    fn program(&mut self, node: AnalyzedProgram<'src>) {
        self.main_fn(node.main_fn);
        for func in node.functions {
            self.function_definition(func);
        }
    }

    fn main_fn(&mut self, body: AnalyzedBlock<'src>) {
        // add to self.functions map (main func idx is always 0)
        self.functions.insert("main", vec![0]);

        // set param_count to 0
        self.param_count = 0;

        // add signature to type section
        self.type_section.push(vec![
            types::FUNC,
            0, // num of params
            0, // num of return vals
        ]);

        // index of type in type_section (main func is always 0)
        self.function_section.push(vec![0]);

        // export main func as WASI `_start` func
        self.export_section.push(
            [
                &[
                    6, // string len
                ][..],
                b"_start", // name of export
                &[
                    0, // export kind (0 = func)
                    0, // index of func in function_section (main func is always 0)
                ],
            ]
            .concat(),
        );

        // function body
        self.function_body(body);
    }

    fn function_definition(&mut self, node: AnalyzedFunctionDefinition<'src>) {
        // don't compile unused functions
        if !node.used {
            return;
        }

        // add to self.functions map
        let func_idx = self.function_section.len();
        self.functions.insert(node.name, func_idx.to_uleb128());

        // clear variable scope
        self.scope.clear();

        // start new buf vor func signature
        let mut buf = vec![types::FUNC];

        // add param types
        let mut params = vec![];
        for (name, type_) in node.params {
            let local_idx = params.len().to_uleb128();
            let wasm_type = utils::type_to_byte(type_);

            if let Some(byte) = wasm_type {
                params.push(byte);
            }
            self.scope.insert(name, wasm_type.map(|t| (t, local_idx)));
        }
        self.param_count = params.len();
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

        // function body
        self.function_body(node.block);
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
                // function budy
                mem::take(&mut self.function_body),
            ]
            .concat(),
        );
    }

    /////////////////////////

    fn statement(&mut self, node: AnalyzedStatement<'src>) {
        match node {
            AnalyzedStatement::Let(node) => self.let_stmt(node),
            AnalyzedStatement::Return(node) => todo!(),
            AnalyzedStatement::Expr(expr) => {
                let expr_type = expr.result_type();
                self.expression(expr);
                if expr_type != Type::Unit {
                    self.function_body.push(instructions::DROP);
                }
            }
        }
    }

    fn let_stmt(&mut self, node: AnalyzedLetStmt<'src>) {
        let wasm_type = utils::type_to_byte(node.expr.result_type());
        let mut local_idx = (self.locals.len() + self.param_count).to_uleb128();
        if let Some(byte) = wasm_type {
            self.locals.push(vec![1, byte]);
        }
        self.scope
            .insert(node.name, wasm_type.map(|t| (t, local_idx.clone())));

        self.expression(node.expr);
        self.function_body.push(instructions::LOCAL_SET);
        self.function_body.append(&mut local_idx);
    }

    fn expression(&mut self, node: AnalyzedExpression<'src>) {
        match node {
            AnalyzedExpression::Block(node) => self.block_expr(*node),
            AnalyzedExpression::If(value) => todo!(),
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
                value.write_uleb128(&mut self.function_body);
            }
            AnalyzedExpression::Char(value) => {
                // `char`s are stored as unsigned `i32`
                self.function_body.push(instructions::I32_CONST);
                value.write_uleb128(&mut self.function_body);
            }
            AnalyzedExpression::Ident(ident) => match &self.scope[ident.ident] {
                Some((_type, local_idx)) => {
                    self.function_body.push(instructions::LOCAL_GET);
                    self.function_body.extend_from_slice(local_idx);
                }
                // unit type requires no instructions
                None => {}
            },
            AnalyzedExpression::Prefix(node) => todo!(),
            AnalyzedExpression::Infix(node) => self.infix_expr(*node),
            AnalyzedExpression::Assign(node) => self.assign_expr(*node),
            AnalyzedExpression::Call(node) => todo!(),
            AnalyzedExpression::Cast(node) => todo!(),
            AnalyzedExpression::Grouped(expr) => self.expression(*expr),
        }
    }

    fn block_expr(&mut self, node: AnalyzedBlock<'src>) {
        for stmt in node.stmts {
            self.statement(stmt);
        }
        if let Some(expr) = node.expr {
            self.expression(expr);
        }
    }

    fn infix_expr(&mut self, node: AnalyzedInfixExpr<'src>) {
        // save type of lhs
        let lhs_type = node.lhs.result_type();

        // compile left and right expression
        self.expression(node.lhs);
        self.expression(node.rhs);

        // match on op and type (analyzer guarantees same type for lhs and rhs)
        let instruction = match (node.op, lhs_type) {
            (InfixOp::Plus, Type::Int) => instructions::I64_ADD,
            (InfixOp::Plus, Type::Float) => instructions::F64_ADD,
            (InfixOp::Minus, Type::Int) => instructions::I64_SUB,
            (InfixOp::Minus, Type::Float) => instructions::F64_SUB,
            (InfixOp::Mul, Type::Int) => instructions::I64_MUL,
            (InfixOp::Mul, Type::Float) => instructions::F64_MUL,
            (InfixOp::Div, Type::Int) => instructions::I64_DIV_S,
            (InfixOp::Div, Type::Float) => instructions::F64_DIV,
            (InfixOp::Rem, Type::Int) => instructions::I64_REM_S,
            (InfixOp::Pow, Type::Int) => todo!(), // TODO: pow
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
            (InfixOp::Gt, Type::Int) => instructions::I32_GT_S,
            (InfixOp::Gt, Type::Float) => instructions::F64_GT,
            (InfixOp::Lte, Type::Int) => instructions::I32_LE_S,
            (InfixOp::Lte, Type::Float) => instructions::F64_LE,
            (InfixOp::Gte, Type::Int) => instructions::I32_GE_S,
            (InfixOp::Gte, Type::Float) => instructions::F64_GE,
            (InfixOp::Shl, Type::Int) => instructions::I64_SHL,
            (InfixOp::Shr, Type::Int) => instructions::I64_SHR_S,
            (InfixOp::BitOr, Type::Int) => instructions::I64_OR,
            (InfixOp::BitOr, Type::Bool) => instructions::I32_OR,
            (InfixOp::BitAnd, Type::Int) => instructions::I64_AND,
            (InfixOp::BitAnd, Type::Bool) => instructions::I32_AND,
            (InfixOp::BitXor, Type::Int) => instructions::I64_XOR,
            (InfixOp::BitXor, Type::Bool) => instructions::I32_XOR,
            // TODO: logical AND and OR
            (InfixOp::And, Type::Bool) => todo!(),
            (InfixOp::Or, Type::Bool) => todo!(),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        };
        self.function_body.push(instruction);
    }

    fn assign_expr(&mut self, node: AnalyzedAssignExpr<'src>) {
        let Some((type_, local_idx)) = &self.scope[node.assignee] else {
            self.expression(node.expr);
            return;
        };

        // get the current value if assignment is more than just `=`
        if node.op != AssignOp::Basic {
            self.function_body.push(instructions::LOCAL_GET);
            self.function_body.extend_from_slice(local_idx);
        }

        // save expr_type
        let expr_type = node.expr.result_type();

        // compile rhs
        self.expression(node.expr);

        // calculate new value for non-basic assignments
        if node.op != AssignOp::Basic {
            // match on op and type (analyzer guarantees same type for variable and expr)
            let instruction = match (node.op, expr_type) {
                (AssignOp::Plus, Type::Int) => instructions::I64_ADD,
                (AssignOp::Plus, Type::Float) => instructions::F64_ADD,
                (AssignOp::Minus, Type::Int) => instructions::I64_SUB,
                (AssignOp::Minus, Type::Float) => instructions::F64_SUB,
                (AssignOp::Mul, Type::Int) => instructions::I64_MUL,
                (AssignOp::Mul, Type::Float) => instructions::F64_MUL,
                (AssignOp::Div, Type::Int) => instructions::I64_DIV_S,
                (AssignOp::Div, Type::Float) => instructions::F64_DIV,
                (AssignOp::Rem, Type::Int) => instructions::I64_REM_S,
                (AssignOp::Pow, Type::Int) => todo!(), // TODO: pow
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

        // set local to new value
        self.function_body.push(instructions::LOCAL_SET);
        self.function_body.extend_from_slice(
            &self.scope[node.assignee]
                .as_ref()
                .expect("we checked this at the top of the function")
                .1,
        );
    }
}

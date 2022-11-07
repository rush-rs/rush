use std::collections::HashMap;

use crate::error::Result;
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    targets::TargetTriple,
    types::BasicMetadataTypeEnum,
    values::BasicValueEnum,
};
use rush_analyzer::{
    ast::{
        AnalyzedBlock, AnalyzedExpression, AnalyzedFunctionDefinition, AnalyzedLetStmt,
        AnalyzedProgram, AnalyzedReturnStmt, AnalyzedStatement,
    },
    Type,
};

pub struct Compiler<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,

    // contains information about the current function
    curr_fn: Option<Function<'ctx>>,
}

#[derive(Debug)]
struct Function<'ctx> {
    // saves the declared variables of the function
    // TODO: remove the need for String allocation
    vars: HashMap<String, BasicValueEnum<'ctx>>,
    // specifies whether the function has already returned
    has_returned: bool,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(context: &'ctx Context, target_triple: TargetTriple) -> Compiler<'ctx> {
        let module = context.create_module("main");

        // setup target machine triple
        module.set_triple(&target_triple);

        Self {
            context,
            module,
            builder: context.create_builder(),
            curr_fn: None,
        }
    }

    pub fn compile(&mut self, program: AnalyzedProgram) -> Result<String> {
        // compile all defined functions which are later used
        for func in program.functions.iter().filter(|func| func.used) {
            self.compile_fn_definition(func);
        }

        // compile the main function
        self.compile_main_fn(&program.main_fn);

        // return the LLVM IR
        println!("{}", self.module.print_to_string().to_string());
        self.module.verify()?;
        Ok(self.module.print_to_string().to_string())
    }

    fn compile_main_fn(&mut self, node: &AnalyzedBlock) {
        // create the main function which returns an int (exit-code)
        let fn_type = self.context.void_type().fn_type(&[], false);
        let main_fn = self
            .module
            .add_function("main", fn_type, Some(Linkage::External));

        // create a new basic block for the main function
        let main_basic_block = self.context.append_basic_block(main_fn, "entry");
        self.builder.position_at_end(main_basic_block);

        // create a new scope for the main function
        self.curr_fn = Some(Function {
            vars: HashMap::new(),
            has_returned: false,
        });

        // build the function's block
        self.compile_block(node);

        dbg!(&self.curr_fn);

        // return exit-code 0 by default
        //let success = self.context.i64_type().const_zero();
        self.build_return(None);
    }

    fn compile_fn_definition(&mut self, node: &AnalyzedFunctionDefinition) {
        // create a new scope for the current function
        self.curr_fn = Some(Function {
            vars: HashMap::new(),
            has_returned: false,
        });

        // create the function's parameters
        let params = node
            .params
            .iter()
            .map(|param| match param.1 {
                Type::Int => BasicMetadataTypeEnum::IntType(self.context.i64_type()),
                Type::Float => BasicMetadataTypeEnum::FloatType(self.context.f64_type()),
                Type::Bool => BasicMetadataTypeEnum::IntType(self.context.bool_type()),
                Type::Char => BasicMetadataTypeEnum::IntType(self.context.i8_type()),
                Type::Unit | Type::Never | Type::Unknown => {
                    unreachable!("the analyzer disallows these types to be used as parameters")
                }
            })
            .collect::<Vec<BasicMetadataTypeEnum>>();

        // create the function's signature
        let signature = match node.return_type {
            Type::Int => self.context.i64_type().fn_type(&params, false),
            Type::Float => self.context.f64_type().fn_type(&params, false),
            Type::Unit => self.context.void_type().fn_type(&params, false),
            Type::Char => self.context.i8_type().fn_type(&params, false),
            Type::Bool => self.context.bool_type().fn_type(&params, false),
            Type::Never => todo!("contemplate what to do on never type functions"),
            Type::Unknown => unreachable!("the function's return type must be known at this point"),
        };

        let function = self
            .module
            .add_function(node.name, signature, Some(Linkage::External));

        // create a new basic block for the function
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        // bind each parameter to the original value (for later reference)
        for (i, param) in node.params.iter().enumerate() {
            let value = function
                .get_nth_param(i as u32)
                .expect("this parameter exists");

            // insert the parameter into the map
            self.curr_fn
                .as_mut()
                .expect("this is only called in function bodies")
                .vars
                .insert(param.0.to_string(), value);
        }

        // build the return value of the function's body
        let return_value = self.compile_block(&node.block);
        self.build_return(return_value);
    }

    /// Compiles a block and returns its return value
    fn compile_block(&mut self, node: &AnalyzedBlock) -> Option<BasicValueEnum<'ctx>> {
        for stmt in &node.stmts {
            if self
                .curr_fn
                .as_ref()
                .expect("this is only called from functions")
                .has_returned
            {
                break;
            }
            self.compile_statement(stmt);
        }
        // if there is an expression, return its value instead of void
        node.expr.as_ref().map(|expr| self.compile_expression(expr))
    }

    fn compile_statement(&mut self, node: &AnalyzedStatement) {
        match node {
            AnalyzedStatement::Let(node) => self.compile_let_statement(node),
            AnalyzedStatement::Return(node) => self.compile_return_statement(node),
            AnalyzedStatement::Expr(node) => {
                self.compile_expression(node);
            }
        }
    }

    fn compile_let_statement(&mut self, node: &AnalyzedLetStmt) {
        // compile the value
        let value = self.compile_expression(&node.expr);
        // insert the value into the vars of the current function
        self.curr_fn
            .as_mut()
            .expect("this is only called from functions")
            .vars
            .insert(node.name.to_string(), value);
    }

    fn compile_return_statement(&mut self, node: &AnalyzedReturnStmt) {
        match node {
            Some(expr) => {
                let value = self.compile_expression(expr);
                self.builder.build_return(Some(&value))
            }
            None => self.builder.build_return(None),
        };
    }

    fn compile_expression(&mut self, node: &AnalyzedExpression) -> BasicValueEnum<'ctx> {
        // TODO: implement expressions
        BasicValueEnum::IntValue(self.context.i64_type().const_int(42, false))
    }

    fn build_return(&mut self, return_value: Option<BasicValueEnum<'ctx>>) {
        let mut curr_fn = self
            .curr_fn
            .as_mut()
            .expect("this is only called from functions");

        if !curr_fn.has_returned {
            curr_fn.has_returned = true;
            match return_value {
                Some(value) => self.builder.build_return(Some(&value)),
                None => self.builder.build_return(None),
            };
        };
    }
}

#[cfg(test)]
mod tests {
    use inkwell::{context::Context, targets::TargetMachine};

    use crate::Compiler;

    #[test]
    fn test_main_fn() {
        let (ast, _) =
            rush_analyzer::analyze("fn aa() -> int { let a = 1; return a; } fn main() { aa(); let a = 1; }").unwrap();

        let context = Context::create();
        let mut compiler = Compiler::new(&context, TargetMachine::get_default_triple());

        let out = compiler.compile(ast).unwrap();
        println!("{out}");
    }
}

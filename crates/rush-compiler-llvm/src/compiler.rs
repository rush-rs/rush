// TODO: go over these points
// - add helper function for getting the current function

use std::collections::HashMap;

use crate::error::Result;
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    targets::TargetTriple,
    types::BasicMetadataTypeEnum,
    values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum},
};
use rush_analyzer::{
    ast::{
        AnalyzedBlock, AnalyzedCallExpr, AnalyzedExpression, AnalyzedFunctionDefinition,
        AnalyzedInfixExpr, AnalyzedLetStmt, AnalyzedPrefixExpr, AnalyzedProgram,
        AnalyzedReturnStmt, AnalyzedStatement,
    },
    InfixOp, PrefixOp, Type,
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

    fn curr_fn(&self) -> &Function<'ctx> {
        self.curr_fn
            .as_ref()
            .expect("this is only called from functions")
    }

    fn curr_fn_mut(&mut self) -> &mut Function<'ctx> {
        self.curr_fn
            .as_mut()
            .expect("this is only called from functions")
    }

    pub fn compile(&mut self, program: AnalyzedProgram) -> Result<String> {
        // compile all defined functions which are later used
        for func in program.functions.iter().filter(|func| func.used) {
            self.compile_fn_definition(func);
        }

        // compile the main function
        self.compile_main_fn(&program.main_fn);

        // return the LLVM IR
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
            self.curr_fn_mut().vars.insert(param.0.to_string(), value);
        }

        // build the return value of the function's body
        let return_value = self.compile_block(&node.block);
        self.build_return(return_value);
    }

    /// Compiles a block and returns its return value
    fn compile_block(&mut self, node: &AnalyzedBlock) -> Option<BasicValueEnum<'ctx>> {
        for stmt in &node.stmts {
            if self.curr_fn().has_returned {
                break;
            }
            self.compile_statement(stmt);
        }
        // if there is an expression, return its value instead of void
        node.expr
            .as_ref()
            .map_or_else(|| None, |expr| self.compile_expression(expr))
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
        self.curr_fn_mut().vars.insert(
            node.name.to_string(),
            value.expect("let bindings with `()` as rhs are forbidden"),
        );
    }

    fn compile_return_statement(&mut self, node: &AnalyzedReturnStmt) {
        match node {
            Some(expr) => {
                let value = self.compile_expression(expr);
                self.build_return(value)
            }
            None => self.build_return(None),
        };
    }

    fn compile_expression(&mut self, node: &AnalyzedExpression) -> Option<BasicValueEnum<'ctx>> {
        match node {
            AnalyzedExpression::Int(value) => Some(BasicValueEnum::IntValue(
                self.context.i64_type().const_int(*value as u64, false),
            )),
            AnalyzedExpression::Ident(name) => Some(
                *self
                    .curr_fn()
                    .vars
                    .get(name.ident)
                    .expect("this variable was declared beforehand"),
            ),
            AnalyzedExpression::Call(node) => self.compile_call_expression(node),
            AnalyzedExpression::Infix(node) => self.compile_infix_expression(node),
            AnalyzedExpression::Prefix(node) => self.compile_prefix_expression(node),
            _ => todo!("implement this expression kind: {:?}", node),
        }
    }

    fn compile_call_expression(&mut self, node: &AnalyzedCallExpr) -> Option<BasicValueEnum<'ctx>> {
        // check that the function is not a builtin one
        let func = match node.func {
            "exit" => {
                let exit_type = self.context.void_type().fn_type(
                    &[BasicMetadataTypeEnum::IntType(self.context.i64_type())],
                    false,
                );
                self.module
                    .add_function("exit", exit_type, Some(Linkage::External))
            }
            // look up the function name inside the module table
            _ => self
                .module
                .get_function(node.func)
                .expect("this is only called in functions"),
        };

        // compile the function args
        let args: Vec<BasicMetadataValueEnum> = node
            .args
            .iter()
            .map(|arg| {
                self.compile_expression(arg)
                    .expect("value of type `()` shall not be used as function parameters")
                    .into()
            })
            .collect();

        let res = self
            .builder
            .build_call(func, &args, format!("ret_{}", node.func).as_str())
            .try_as_basic_value();

        if res.is_left() {
            Some(res.unwrap_left())
        } else {
            None
        }
    }

    fn compile_infix_expression(
        &mut self,
        node: &AnalyzedInfixExpr,
    ) -> Option<BasicValueEnum<'ctx>> {
        let lhs = self
            .compile_expression(&node.lhs)
            .expect("can only use infix expressions on non-unit values");

        let rhs = self
            .compile_expression(&node.rhs)
            .expect("can only use infix expressions on non-unit values");

        match node.result_type {
            Type::Int => {
                let lhs = lhs.into_int_value();
                let rhs = rhs.into_int_value();
                Some(
                    match node.op {
                        InfixOp::Plus => self.builder.build_int_add(lhs, rhs, "sum"),
                        InfixOp::Mul => self.builder.build_int_mul(lhs, rhs, "prod"),
                        InfixOp::Div => self.builder.build_int_signed_div(lhs, rhs, "res"),
                        InfixOp::Rem => self.builder.build_int_signed_rem(lhs, rhs, "rem"),
                        InfixOp::Pow => todo!(),
                        _ => todo!(),
                    }
                    .as_basic_value_enum(),
                )
            }
            _ => todo!(),
        }
    }

    fn compile_prefix_expression(&mut self, node: &AnalyzedPrefixExpr) -> BasicValueEnum<'ctx> {
        let base = self
            .compile_expression(&node.expr)
            .expect("this must return a non `()` value");

        match (node.expr.result_type(), node.op) {
            (Type::Int, PrefixOp::Neg) => self
                .builder
                .build_int_neg(base.into_int_value(), "neg")
                .as_basic_value_enum(),
            (Type::Float, PrefixOp::Neg) => self
                .builder
                .build_float_neg(base.into_float_value(), "neg")
                .as_basic_value_enum(),
            // TODO: is this the right way of negating bools?
            (Type::Bool, PrefixOp::Not) => self
                .builder
                .build_int_neg(base.into_int_value(), "neg")
                .as_basic_value_enum(),
            _ => unreachable!("other types cannot be negated"),
        }
    }

    fn build_return(&mut self, return_value: Option<BasicValueEnum<'ctx>>) {
        let mut curr_fn = self.curr_fn_mut();
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
    use std::fs;

    use inkwell::{context::Context, targets::TargetMachine};

    use crate::Compiler;

    #[test]
    fn test_main_fn() {
        let (ast, _) = rush_analyzer::analyze(
            "
        fn exit_me(code: int) {
            exit(code * 2);
        }

        fn main() {
            exit_me(-2);
        }
        ",
        )
        .unwrap();

        let context = Context::create();
        let mut compiler = Compiler::new(&context, TargetMachine::get_default_triple());

        let out = compiler.compile(ast).unwrap();
        fs::write("./llvm_ir_test/main.ll", &out).unwrap();
        println!("{out}");
    }
}

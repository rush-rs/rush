// TODO: go over these points
// - add helper function for getting the current function

use std::collections::{HashMap, HashSet};

use crate::error::Result;
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    targets::TargetTriple,
    types::BasicMetadataTypeEnum,
    values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum, FloatValue, PointerValue},
    FloatPredicate, IntPredicate,
};
use rush_analyzer::{
    ast::{
        AnalyzedAssignExpr, AnalyzedBlock, AnalyzedCallExpr, AnalyzedCastExpr, AnalyzedExpression,
        AnalyzedFunctionDefinition, AnalyzedInfixExpr, AnalyzedLetStmt, AnalyzedPrefixExpr,
        AnalyzedProgram, AnalyzedReturnStmt, AnalyzedStatement,
    },
    InfixOp, PrefixOp, Type,
};

pub struct Compiler<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,

    // contains information about the current function
    curr_fn: Option<Function<'ctx>>,

    // contains all builtin functions already declared in the IR
    declared_builtins: HashSet<&'ctx str>,
}

#[derive(Debug)]
struct Function<'ctx> {
    // saves the declared variables of the function
    // TODO: remove the need for String allocation
    vars: HashMap<String, PointerValue<'ctx>>,
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
            declared_builtins: HashSet::new(),
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
        let params: Vec<BasicMetadataTypeEnum> = node
            .params
            .iter()
            .filter_map(|param| {
                Some(match param.1 {
                    Type::Int => BasicMetadataTypeEnum::IntType(self.context.i64_type()),
                    Type::Float => BasicMetadataTypeEnum::FloatType(self.context.f64_type()),
                    Type::Bool => BasicMetadataTypeEnum::IntType(self.context.bool_type()),
                    Type::Char => BasicMetadataTypeEnum::IntType(self.context.i8_type()),
                    Type::Unit => {
                        // allocate a pointer for the dummy parameter
                        let ptr = self
                            .builder
                            .build_alloca(self.context.i8_type(), "skipped_unit");
                        // insert the dummy pointer into vars
                        self.curr_fn_mut().vars.insert(param.0.to_string(), ptr);
                        return None;
                    }
                    Type::Never | Type::Unknown => {
                        unreachable!("the analyzer disallows these types to be used as parameters")
                    }
                })
            })
            .collect();

        // create the function's signature
        let signature = match node.return_type {
            Type::Int => self.context.i64_type().fn_type(&params, false),
            Type::Float => self.context.f64_type().fn_type(&params, false),
            Type::Unit => self.context.void_type().fn_type(&params, false),
            Type::Char => self.context.i8_type().fn_type(&params, false),
            Type::Bool => self.context.bool_type().fn_type(&params, false),
            Type::Unknown | Type::Never => {
                unreachable!("functions do not return values of these types")
            }
        };

        let function = self
            .module
            .add_function(node.name, signature, Some(Linkage::External));

        // create a new basic block for the function
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        // bind each non-unit-type parameter to the original value (for later reference)
        for (i, param) in node
            .params
            .iter()
            .filter(|param| param.1 != Type::Unit) // filter out any values of type `()`
            .enumerate()
        {

            // todo: do this

            // allocate a pointer for the parameters
            let ptr = self.builder.build_alloca(ty, name)
            
            let value = function
                .get_nth_param(i as u32)
                .expect("this parameter exists");

            // insert the parameter into the vars map
            self.curr_fn_mut()
                .vars
                .insert(param.0.to_string(), Some(value));
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

        // allocate a pointer to the value
        let ptr = self
            .builder
            .build_alloca(self.context.i64_type(), node.name);

        // insert the value into the vars of the current function
        self.curr_fn_mut().vars.insert(node.name.to_string(), value);
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
            AnalyzedExpression::Float(value) => Some(BasicValueEnum::FloatValue(
                self.context.f64_type().const_float(*value),
            )),
            AnalyzedExpression::Char(value) => Some(BasicValueEnum::IntValue(
                self.context.i8_type().const_int(*value as u64, false),
            )),
            AnalyzedExpression::Bool(value) => Some(BasicValueEnum::IntValue(
                self.context.i8_type().const_int(u64::from(*value), false),
            )),
            AnalyzedExpression::Ident(name) => Some(
                self.curr_fn()
                    .vars
                    .get(name.ident)
                    .expect("this variable was declared beforehand")
                    .as_basic_value_enum(),
            ),
            AnalyzedExpression::Call(node) => self.compile_call_expression(node),
            AnalyzedExpression::Grouped(node) => self.compile_expression(node),
            AnalyzedExpression::Block(node) => self.compile_block(node),
            AnalyzedExpression::Infix(node) => Some(self.compile_infix_expression(node)),
            AnalyzedExpression::Prefix(node) => Some(self.compile_prefix_expression(node)),
            AnalyzedExpression::Cast(node) => Some(self.compile_cast_expression(node)),
            AnalyzedExpression::Assign(node) => {
                self.compile_assign_expression(node);
                None
            }
            AnalyzedExpression::If(_) => todo!("if still needs to be implemented"),
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
                // declare the exit function if it does not exist
                match self.declared_builtins.insert("exit") {
                    true => self
                        .module
                        .add_function("exit", exit_type, Some(Linkage::External)),
                    false => self
                        .module
                        .get_function("exit")
                        .expect("exit was previously declared"),
                }
            }
            // look up the function name inside the module table
            _ => self
                .module
                .get_function(node.func)
                .expect("this is only called in functions"),
        };

        // compile the function args excluding any unit-type values
        let args: Vec<BasicMetadataValueEnum> = node
            .args
            .iter()
            .filter_map(|arg| {
                self.compile_expression(arg)
                    .map(BasicMetadataValueEnum::from)
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

    fn compile_infix_expression(&mut self, node: &AnalyzedInfixExpr) -> BasicValueEnum<'ctx> {
        let lhs = self
            .compile_expression(&node.lhs)
            .expect("can only use infix expressions on non-unit values");

        let rhs = self
            .compile_expression(&node.rhs)
            .expect("can only use infix expressions on non-unit values");

        match node.lhs.result_type() {
            Type::Float => {
                let lhs = lhs.into_float_value();
                let rhs = rhs.into_float_value();
                match node.op {
                    // arithmetic operators
                    InfixOp::Plus => self.builder.build_float_add(lhs, rhs, "f_sum"),
                    InfixOp::Minus => self.builder.build_float_sub(lhs, rhs, "f_sum"),
                    InfixOp::Mul => self.builder.build_float_mul(lhs, rhs, "f_prod"),
                    InfixOp::Div => self.builder.build_float_div(lhs, rhs, "f_prod"),
                    InfixOp::Rem => self.builder.build_float_rem(lhs, rhs, "f_rem"),
                    InfixOp::Pow => self.pow_helper(lhs, rhs),
                    // comparison operators
                    op => {
                        let (op, label) = match op {
                            InfixOp::Eq => (FloatPredicate::OEQ, "f_eq"),
                            InfixOp::Neq => (FloatPredicate::ONE, "f_neq"),
                            InfixOp::Gt => (FloatPredicate::OGT, "f_gt"),
                            InfixOp::Gte => (FloatPredicate::OGE, "f_gte"),
                            InfixOp::Lt => (FloatPredicate::OLT, "f_lt"),
                            InfixOp::Lte => (FloatPredicate::OLE, "f_lte"),
                            _ => unreachable!("other operators cannot be used on float"),
                        };
                        return self
                            .builder
                            .build_float_compare(op, lhs, rhs, label)
                            .as_basic_value_enum();
                    }
                }
                .as_basic_value_enum()
            }
            Type::Int => {
                let lhs = lhs.into_int_value();
                let rhs = rhs.into_int_value();
                match node.op {
                    // arithmetic operators
                    InfixOp::Plus => self.builder.build_int_add(lhs, rhs, "i_sum"),
                    InfixOp::Minus => self.builder.build_int_sub(lhs, rhs, "i_sum"),
                    InfixOp::Mul => self.builder.build_int_mul(lhs, rhs, "i_prod"),
                    InfixOp::Div => self.builder.build_int_signed_div(lhs, rhs, "i_prod"),
                    InfixOp::Rem => self.builder.build_int_signed_rem(lhs, rhs, "i_rem"),
                    InfixOp::Pow => {
                        // convert both arguments to f64
                        let lhs_f64 = self.builder.build_signed_int_to_float(
                            lhs,
                            self.context.f64_type(),
                            "pow_lhs",
                        );
                        let rhs_f64 = self.builder.build_signed_int_to_float(
                            rhs,
                            self.context.f64_type(),
                            "pow_rhs",
                        );

                        let pow_res = self.pow_helper(lhs_f64, rhs_f64);

                        // convert the result back to i64
                        let res = self.builder.build_float_to_signed_int(
                            pow_res,
                            self.context.i64_type(),
                            "pow_i64_res",
                        );

                        res
                    }
                    // bitwise operators
                    InfixOp::Shl => self.builder.build_left_shift(lhs, rhs, "i_shl"),
                    InfixOp::Shr => self.builder.build_right_shift(lhs, rhs, true, "i_shr"),
                    InfixOp::BitOr => self.builder.build_or(lhs, rhs, "i_bor"),
                    InfixOp::BitAnd => self.builder.build_and(lhs, rhs, "i_band"),
                    InfixOp::BitXor => self.builder.build_xor(lhs, rhs, "i_bxor"),
                    // comparison operators
                    op => {
                        let (op, label) = match op {
                            InfixOp::Eq => (IntPredicate::EQ, "i_eq"),
                            InfixOp::Neq => (IntPredicate::NE, "i_neq"),
                            InfixOp::Gt => (IntPredicate::SGT, "i_gt"),
                            InfixOp::Gte => (IntPredicate::SGE, "i_gte"),
                            InfixOp::Lt => (IntPredicate::SLT, "i_lt"),
                            InfixOp::Lte => (IntPredicate::SLE, "i_lte"),
                            _ => unreachable!("other operators cannot be used on int"),
                        };
                        return self
                            .builder
                            .build_int_compare(op, lhs, rhs, label)
                            .as_basic_value_enum();
                    }
                }
                .as_basic_value_enum()
            }
            Type::Char => {
                let lhs = lhs.into_int_value();
                let rhs = rhs.into_int_value();

                match node.op {
                    InfixOp::Eq => {
                        self.builder
                            .build_int_compare(IntPredicate::EQ, lhs, rhs, "c_eq")
                    }
                    InfixOp::Neq => {
                        self.builder
                            .build_int_compare(IntPredicate::EQ, lhs, rhs, "c_eq")
                    }
                    _ => unreachable!("other operators cannot be used on char"),
                }
                .as_basic_value_enum()
            }
            Type::Bool => {
                let lhs = lhs.into_int_value();
                let rhs = rhs.into_int_value();

                match node.op {
                    InfixOp::Eq => {
                        self.builder
                            .build_int_compare(IntPredicate::EQ, lhs, rhs, "b_eq")
                    }
                    InfixOp::Neq => {
                        self.builder
                            .build_int_compare(IntPredicate::NE, lhs, rhs, "b_neq")
                    }
                    InfixOp::BitOr | InfixOp::Or => self.builder.build_or(lhs, rhs, "b_or"),
                    InfixOp::BitAnd | InfixOp::And => self.builder.build_and(lhs, rhs, "b_and"),
                    InfixOp::BitXor => self.builder.build_xor(lhs, rhs, "b_xor"),
                    _ => unreachable!("other operators cannot be used on bool"),
                }
                .as_basic_value_enum()
            }
            _ => todo!("implement other types: {:?}", node.result_type),
        }
    }

    fn pow_helper(&mut self, lhs: FloatValue<'ctx>, rhs: FloatValue<'ctx>) -> FloatValue<'ctx> {
        // declare the pow builtin function if not already declared
        if self.declared_builtins.insert("pow") {
            let pow_type = self.context.f64_type().fn_type(
                &[
                    BasicMetadataTypeEnum::FloatType(self.context.f64_type()),
                    BasicMetadataTypeEnum::FloatType(self.context.f64_type()),
                ],
                false,
            );
            self.module
                .add_function("pow", pow_type, Some(Linkage::External));
        }

        // call the pow builtin function
        let args: Vec<BasicMetadataValueEnum> = vec![
            BasicValueEnum::FloatValue(lhs).into(),
            BasicValueEnum::FloatValue(rhs).into(),
        ];

        let res = self
            .builder
            .build_call(
                self.module
                    .get_function("pow")
                    .expect("pow is already declared"),
                &args,
                "pow",
            )
            .try_as_basic_value()
            .expect_left("pow always returns a value");

        res.into_float_value()
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
            (Type::Bool, PrefixOp::Not) => {
                // convert the original type to i1
                let value = self.builder.build_int_cast(
                    base.into_int_value(),
                    self.context.bool_type(),
                    "bool_tmp",
                );
                // use XOR to do a negate operation on the bool
                let negated = self.builder.build_xor(
                    value,
                    self.context.bool_type().const_int(1, false),
                    "bool_neg",
                );
                // return the negated bool
                negated.as_basic_value_enum()
            }
            _ => unreachable!("other types cannot be negated"),
        }
    }

    fn compile_cast_expression(&mut self, node: &AnalyzedCastExpr) -> BasicValueEnum<'ctx> {
        let lhs = self
            .compile_expression(&node.expr)
            .expect("casting is only possible on non-unit values");

        match (lhs, node.type_) {
            // required if a char is casted as an int
            (BasicValueEnum::IntValue(lhs_int), Type::Int) => self
                .builder
                .build_int_cast(lhs_int, self.context.i64_type(), "ii_cast")
                .as_basic_value_enum(),
            (BasicValueEnum::IntValue(lhs_int), Type::Float) => self
                .builder
                .build_signed_int_to_float(lhs_int, self.context.f64_type(), "if_cast")
                .as_basic_value_enum(),
            (BasicValueEnum::IntValue(lhs_int), Type::Char) => self
                .builder
                .build_int_cast(lhs_int, self.context.i8_type(), "ic_cast")
                .as_basic_value_enum(),
            (BasicValueEnum::FloatValue(_), Type::Float) => lhs,
            (BasicValueEnum::FloatValue(lhs_float), Type::Int) => self
                .builder
                .build_float_to_signed_int(lhs_float, self.context.i64_type(), "fi_cast")
                .as_basic_value_enum(),
            (BasicValueEnum::FloatValue(lhs_float), Type::Char) => self
                .builder
                .build_float_to_unsigned_int(lhs_float, self.context.i8_type(), "fc_cast")
                .as_basic_value_enum(),
            _ => todo!(""),
        }
    }

    fn compile_assign_expression(&mut self, node: &AnalyzedAssignExpr) {
        todo!()
    }

    /// Builds a return instruction if the current block has no terminater
    /// TODO: impl block check
    fn build_return(&mut self, return_value: Option<BasicValueEnum<'ctx>>) {
        let mut curr_fn = self.curr_fn_mut();
        if !curr_fn.has_returned {
            curr_fn.has_returned = true;
            match return_value {
                Some(value) => self.builder.build_return(Some(&value)),
                None => self.builder.build_return(None),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use inkwell::{context::Context, targets::TargetMachine};

    use crate::Compiler;

    #[test]
    fn test_main_fn() {
        let (ast, _) = rush_analyzer::analyze(include_str!("../test.rush")).unwrap();

        let context = Context::create();
        let mut compiler = Compiler::new(&context, TargetMachine::get_default_triple());

        let out = compiler.compile(ast).unwrap();
        fs::write("./main.ll", &out).unwrap();
        println!("{out}");
    }
}

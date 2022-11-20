use inkwell::{
    types::BasicMetadataTypeEnum,
    values::{BasicMetadataValueEnum, BasicValueEnum, FloatValue, IntValue},
    FloatPredicate, IntPredicate,
};

use crate::Compiler;

impl<'ctx> Compiler<'ctx> {
    /// Helper function for the `**` and `**=` operators.
    /// Because LLVM does not support the pow instruction, the corelib is used.
    /// This function will declare the `pow` function if not already done previously.
    /// The `pow` function is then called using the `lhs` and `rhs` arguments.
    pub(crate) fn __rush_internal_pow(
        &mut self,
        lhs: IntValue<'ctx>,
        rhs: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        // declare the pow builtin function if not already declared
        if self.declared_builtins.insert("core::pow") {
            self.declare_rush_internal_pow();
        }

        let args: Vec<BasicMetadataValueEnum> = vec![
            BasicValueEnum::IntValue(lhs).into(),
            BasicValueEnum::IntValue(rhs).into(),
        ];

        // call the pow builtin function
        let res = self
            .builder
            .build_call(
                self.module
                    .get_function("core::pow")
                    .expect("pow is declared above"),
                &args,
                "pow_res",
            )
            .try_as_basic_value()
            .expect_left("pow always returns a value");

        res.into_int_value()
    }

    /// Declares a rush core function for calculating the power of integer values.
    /// fn pow(base: int, mut exp: int) -> int {
    ///     let mut res = 1;
    ///     if exp < 0 {
    ///         return 0;
    ///     }
    ///     while exp != 0 {
    ///         res *= base;
    ///         exp -= 1;
    ///     }
    ///     res
    /// }
    pub(crate) fn declare_rush_internal_pow(&mut self) {
        // save the basic block to jump to when done
        // this is required because the function is generated when needed
        let origin_block = self
            .builder
            .get_insert_block()
            .expect("there is a bb before this");

        let i64 = self.context.i64_type();

        let pow_type = i64.fn_type(
            &[
                BasicMetadataTypeEnum::IntType(i64),
                BasicMetadataTypeEnum::IntType(i64),
            ],
            false,
        );

        let function = self.module.add_function("core::pow", pow_type, None);

        // add basic blocks for the function
        let fn_body = self.context.append_basic_block(function, "entry");
        let return_0_block = self.context.append_basic_block(function, "ret_0");

        // add basic blocks for the loop
        let loop_head = self.context.append_basic_block(function, "loop_head");
        let loop_body = self.context.append_basic_block(function, "loop_body");
        let after_loop = self.context.append_basic_block(function, "after_loop");

        //// Before loop ////
        self.builder.position_at_end(fn_body);

        // get the base and exp from the args
        let base = function.get_params()[0].into_int_value();
        let exp = function.get_params()[1];
        let exp_ptr = self.builder.build_alloca(i64, "exp_ptr");
        self.builder.build_store(exp_ptr, exp);

        // create the accumulator variable
        self.builder.position_at_end(fn_body);
        let acc_ptr = self.builder.build_alloca(i64, "accumulator");
        self.builder.build_store(acc_ptr, i64.const_int(1, false));

        //// Edge case: exp < 0 ////
        let exp_lt_0 = self.builder.build_int_compare(
            IntPredicate::SLT,
            exp.into_int_value(),
            i64.const_zero(),
            "exp_lt_0",
        );
        self.builder
            .build_conditional_branch(exp_lt_0, return_0_block, loop_head);
        self.builder.position_at_end(return_0_block);
        self.builder.build_return(Some(&i64.const_zero()));

        //// Loop head ////
        self.builder.position_at_end(loop_head);

        // if the exponent is 0, quit the loop
        let exp = self.builder.build_load(exp_ptr, "exp").into_int_value();
        let break_cond =
            self.builder
                .build_int_compare(IntPredicate::EQ, exp, i64.const_zero(), "break_cond");
        self.builder
            .build_conditional_branch(break_cond, after_loop, loop_body);

        //// Loop body ////
        self.builder.position_at_end(loop_body);

        // accumulator *= base
        let acc = self.builder.build_load(acc_ptr, "acc").into_int_value();
        let acc_mul_ass = self.builder.build_int_mul(acc, base, "acc_mul_ass");
        self.builder.build_store(acc_ptr, acc_mul_ass);

        // exp -= 1
        let exp = self.builder.build_load(exp_ptr, "exp").into_int_value();
        let exp_sub_ass = self
            .builder
            .build_int_sub(exp, i64.const_int(1, false), "exp_sub_ass");
        self.builder.build_store(exp_ptr, exp_sub_ass);

        // repeat iteration (jump to loop head)
        self.builder.build_unconditional_branch(loop_head);

        //// After loop ////
        self.builder.position_at_end(after_loop);
        let res = self.builder.build_load(acc_ptr, "res");
        self.builder.build_return(Some(&res));

        // jump back to the origin basic block
        self.builder.position_at_end(origin_block);
    }

    pub(crate) fn __rush_internal_cast_float_to_char(
        &mut self,
        src: FloatValue<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        // declare the `core::float_to_char` function if not declared already
        if self.declared_builtins.insert("core::float_to_char") {
            self.declare_rush_internal_cast_float_to_char()
        }
        let func = self
            .module
            .get_function("core::float_to_char")
            .expect("declared above");

        let args = vec![BasicMetadataValueEnum::from(src)];

        // call the function with the `src` as the argument
        let res = self
            .builder
            .build_call(func, &args, "fc_cast")
            .try_as_basic_value();

        res.expect_left("always returns i8")
    }

    /// Defines the builtin function responsible for converting a float into a char
    /// If the float is < 0.0, the char is 0.
    /// Otherwise, if the float is > 127.0, the char is 127.
    /// If the float is 0.0 < f < 127, the char is the truncated value of the float
    pub(crate) fn declare_rush_internal_cast_float_to_char(&mut self) {
        // save the basic block to jump to when done
        // this is required because the function is generated when needed
        let origin_block = self
            .builder
            .get_insert_block()
            .expect("there is a bb before this");

        // define the function signature
        let params = vec![BasicMetadataTypeEnum::FloatType(self.context.f64_type())];
        let signature = self.context.i8_type().fn_type(&params, false);
        let function = self
            .module
            .add_function("core::float_to_char", signature, None);

        // create a new basic block for the function
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        // get the first argument's value from the function
        let from_value = function.get_params()[0];

        // check if the from value is greater than 127
        // in this case, the result is 127 i8
        let gt_127_then_block = self.context.append_basic_block(function, "gt_127");
        let is_lt_0_block = self.context.append_basic_block(function, "lt_0_check");

        // add the condition ( arg > 127 )
        let is_gt_127 = self.builder.build_float_compare(
            FloatPredicate::OGT,
            from_value.into_float_value(),
            self.context.f64_type().const_float(127.0),
            "gt_127_cond",
        );

        // add the conditional jump to the `gt_127_then_block` block
        self.builder
            .build_conditional_branch(is_gt_127, gt_127_then_block, is_lt_0_block);

        // if the value is > 127, return 127 i8
        self.builder.position_at_end(gt_127_then_block);
        let const_127 = self.context.i8_type().const_int(127, false);
        self.builder.build_return(Some(&const_127));

        // otherwise check if the value is less than 0
        // in this case, the result is 0 i8
        self.builder.position_at_end(is_lt_0_block);

        let lt_0_then_block = self.context.append_basic_block(function, "lt_0");
        let merge_block = self.context.append_basic_block(function, "merge");

        // add the condition ( arg < 0 )
        let is_lt_0 = self.builder.build_float_compare(
            FloatPredicate::OLT,
            from_value.into_float_value(),
            self.context.f64_type().const_zero(),
            "lt_0_cond",
        );

        // add the conditional jump to the `lt_0_then_block` block
        self.builder
            .build_conditional_branch(is_lt_0, lt_0_then_block, merge_block);

        // if the value is < 0, return 0
        self.builder.position_at_end(lt_0_then_block);
        let const_0 = self.context.i8_type().const_int(0, false);
        self.builder.build_return(Some(&const_0));

        // the final merge block just returns the truncated input
        self.builder.position_at_end(merge_block);

        // truncate the input to i8
        let char_res = self.builder.build_float_to_signed_int(
            from_value.into_float_value(),
            self.context.i8_type(),
            "char_res",
        );
        // return the char result in the end
        self.builder.build_return(Some(&char_res));

        // jump back to the origin basic block
        self.builder.position_at_end(origin_block);
    }

    pub(crate) fn __rush_internal_cast_int_to_char(
        &mut self,
        src: IntValue<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        // declare the `core::int_to_char` function if not declared already
        if self.declared_builtins.insert("core::int_to_char") {
            self.declare_rush_internal_int_to_char()
        }
        let func = self
            .module
            .get_function("core::int_to_char")
            .expect("declared above");

        let args = vec![BasicMetadataValueEnum::from(src)];

        // call the function with the `src` as the argument
        let res = self
            .builder
            .build_call(func, &args, "ic_cast")
            .try_as_basic_value();

        res.expect_left("always returns i8")
    }

    /// Defines the builtin function responsible for converting an int into a char
    pub(crate) fn declare_rush_internal_int_to_char(&mut self) {
        // save the basic block to jump to when done
        let origin_block = self
            .builder
            .get_insert_block()
            .expect("there must be a bb before this");

        // define the function signature
        let params = vec![BasicMetadataTypeEnum::IntType(self.context.i64_type())];
        let signature = self.context.i8_type().fn_type(&params, false);
        let function = self
            .module
            .add_function("core::int_to_char", signature, None);

        // create a new basic block for the function
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        // get the first argument's value from the function
        let from_value = function.get_params()[0];

        // check if the from value is greater than 127
        // in this case, the result is 127 i8
        let gt_127_then_block = self.context.append_basic_block(function, "gt_127");
        let is_lt_0_block = self.context.append_basic_block(function, "lt_0_check");

        // add the condition ( arg > 127 )
        let is_gt_127 = self.builder.build_int_compare(
            IntPredicate::SGT,
            from_value.into_int_value(),
            self.context.i64_type().const_int(127, true),
            "gt_127_cond",
        );

        // add the conditional jump to the `gt_127_then_block` block
        self.builder
            .build_conditional_branch(is_gt_127, gt_127_then_block, is_lt_0_block);

        // if the value is > 127, return 127 i8
        self.builder.position_at_end(gt_127_then_block);
        let const_127 = self.context.i8_type().const_int(127, false);
        self.builder.build_return(Some(&const_127));

        // otherwise check if the value is less than 0
        // in this case, the result is 0 i8
        self.builder.position_at_end(is_lt_0_block);

        let lt_0_then_block = self.context.append_basic_block(function, "lt_0");
        let merge_block = self.context.append_basic_block(function, "merge");

        // add the condition ( arg < 0 )
        let is_lt_0 = self.builder.build_int_compare(
            IntPredicate::SLT,
            from_value.into_int_value(),
            self.context.i64_type().const_zero(),
            "lt_0_cond",
        );

        // add the conditional jump to the `lt_0_then_block` block
        self.builder
            .build_conditional_branch(is_lt_0, lt_0_then_block, merge_block);

        // if the value is < 0, return 0
        self.builder.position_at_end(lt_0_then_block);
        let const_0 = self.context.i8_type().const_int(0, false);
        self.builder.build_return(Some(&const_0));

        // the final merge block just returns the truncated input
        self.builder.position_at_end(merge_block);

        // truncate the input to i8
        let char_res = self.builder.build_int_cast(
            from_value.into_int_value(),
            self.context.i8_type(),
            "char_res",
        );
        // return the char result in the end
        self.builder.build_return(Some(&char_res));

        // jump back to the origin basic block
        self.builder.position_at_end(origin_block);
    }
}

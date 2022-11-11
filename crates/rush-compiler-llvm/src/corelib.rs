use inkwell::{
    module::Linkage,
    types::BasicMetadataTypeEnum,
    values::{BasicMetadataValueEnum, BasicValueEnum, FloatValue},
    FloatPredicate, IntPredicate,
};

use crate::Compiler;

impl<'ctx> Compiler<'ctx> {
    /// Helper function for the `**` and `**=` operators.
    /// Because LLVM does not support the pow instruction, the GLIBC `pow` function is used.
    /// This function will declare the `pow` function if not already done previously.
    /// The `pow` function is then called using the `lhs` and `rhs` arguments.
    pub(crate) fn invoke_pow(
        &mut self,
        lhs: FloatValue<'ctx>,
        rhs: FloatValue<'ctx>,
    ) -> FloatValue<'ctx> {
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

        let args: Vec<BasicMetadataValueEnum> = vec![
            BasicValueEnum::FloatValue(lhs).into(),
            BasicValueEnum::FloatValue(rhs).into(),
        ];

        // call the pow builtin function
        let res = self
            .builder
            .build_call(
                self.module
                    .get_function("pow")
                    .expect("pow is declared above"),
                &args,
                "pow",
            )
            .try_as_basic_value()
            .expect_left("pow always returns a value");

        res.into_float_value()
    }

    /// Defines the builtin function responsible for converting a float into a char
    /// If the float is < 0.0, the char is 0.
    /// Otherwise, if the float is > 127.0, the char is 127.
    /// If the float is 0.0 < f < 127, the char is the truncated value of the float
    pub(crate) fn declare_core_float_to_char(&mut self) {
        // save the basic block to jump to when done
        // this is required because the function is generated when needed
        let origin_block = self
            .builder
            .get_insert_block()
            .expect("there is a bb before this");
        // define the function signature
        let params = vec![BasicMetadataTypeEnum::FloatType(self.context.f64_type())];
        let signature = self.context.i8_type().fn_type(&params, false);
        let function =
            self.module
                .add_function("core_float_to_char", signature, Some(Linkage::External));

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

    /// Defines the builtin function responsible for converting an int into a char
    pub(crate) fn define_core_int_to_char(&mut self) {
        // save the basic block to jump to when done
        let origin_block = self
            .builder
            .get_insert_block()
            .expect("there must be a bb before this");
        // define the function signature
        let params = vec![BasicMetadataTypeEnum::IntType(self.context.i64_type())];
        let signature = self.context.i8_type().fn_type(&params, false);
        let function =
            self.module
                .add_function("core_int_to_char", signature, Some(Linkage::External));

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

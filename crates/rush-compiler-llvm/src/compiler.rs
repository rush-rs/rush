use std::collections::{HashMap, HashSet};

use crate::{error::Result, Error};
use inkwell::{
    basic_block::BasicBlock,
    builder::Builder,
    context::Context,
    memory_buffer::MemoryBuffer,
    module::{Linkage, Module},
    passes::{PassManager, PassManagerBuilder},
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple},
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum},
    values::{
        BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, InstructionOpcode,
        PointerValue,
    },
    FloatPredicate, IntPredicate, OptimizationLevel,
};
use rush_analyzer::{ast::*, AssignOp, InfixOp, PrefixOp, Type};

pub struct Compiler<'ctx, 'src> {
    // inkwell components
    pub(crate) context: &'ctx Context,
    pub(crate) module: Module<'ctx>,
    pub(crate) builder: Builder<'ctx>,
    // contains information about the current function
    pub(crate) curr_fn: Option<Function<'ctx, 'src>>,
    // contains necessary metadata about current loops
    // the last element is the most inner loop
    pub(crate) loops: Vec<Loop<'ctx>>,
    // the scope stack (first is the root / function scope, last is the current scope)
    pub(crate) scopes: Vec<HashMap<&'src str, Variable<'ctx>>>,
    // global variables
    pub(crate) globals: HashMap<&'src str, PointerValue<'ctx>>,
    // a set of all builtin functions already declared (`imported`) so far
    pub(crate) declared_builtins: HashSet<&'ctx str>,
    // specifies the target machine
    pub(crate) target_triple: TargetTriple,
    // specifies the optimization level
    pub(crate) optimization: OptimizationLevel,
}

pub(crate) struct Loop<'ctx> {
    // saves the loop_start basic block
    loop_head: BasicBlock<'ctx>,
    // saves the after_loop basic block
    after_loop: BasicBlock<'ctx>,
    // contains the allocations available for the loop
    allocations: Vec<(String, Type, PointerValue<'ctx>)>,
}

pub(crate) struct Function<'ctx, 'src> {
    // specifies the name of the function
    name: &'src str,
    // holds the LLVM function value
    llvm_value: FunctionValue<'ctx>,
}

#[derive(Clone, Copy)]
pub(crate) enum Variable<'ctx> {
    /// A mutable variable which is assigned to
    Mut(PointerValue<'ctx>),
    /// A static variable which is only declared and used
    Const(BasicValueEnum<'ctx>),
}

impl<'ctx, 'src> Compiler<'ctx, 'src> {
    /// Creates and returns a new [`Compiler`].
    /// Requires a new [`Context`] to be used by the compiler.
    /// The LLVM backend can be specified using a [`TargetTriple`].
    /// The optimization level is given through a [`OptimizationLevel`]
    pub fn new(
        context: &'ctx Context,
        target_triple: TargetTriple,
        optimization: OptimizationLevel,
    ) -> Compiler<'ctx, 'src> {
        let module = context.create_module("main");
        // setup target machine triple
        module.set_triple(&target_triple);
        Self {
            context,
            module,
            builder: context.create_builder(),
            curr_fn: None,
            scopes: vec![],
            globals: HashMap::new(),
            loops: vec![],
            declared_builtins: HashSet::new(),
            target_triple,
            optimization,
        }
    }

    // Returns a mutable reference to the current scope
    /// Panics if there are no scopes.
    fn scope_mut(&mut self) -> &mut HashMap<&'src str, Variable<'ctx>> {
        let len = self.scopes.len();
        &mut self.scopes[len - 1]
    }

    /// Helper function for accessing the current function
    /// Can panic when called from outside of functions
    fn curr_fn(&self) -> &Function<'ctx, 'src> {
        self.curr_fn
            .as_ref()
            .expect("this is only called from functions")
    }

    /// Compiles the given [`AnalyzedProgram`] to object code and the LLVM IR.
    /// Errors can occur if the target triple is invalid or the code generation fails
    /// The `main_fn` param specifies whether the entry is the main function or `_start`.
    pub fn compile(
        &mut self,
        program: &'src AnalyzedProgram,
        main_fn: bool,
    ) -> Result<(MemoryBuffer, String)> {
        // declare all global variables
        for global in &program.globals {
            self.declare_global(global.name, &global.expr);
        }

        // add all function signatures beforehand
        for func in program.functions.iter().filter(|func| func.used) {
            self.compile_fn_signature(func);
        }

        // compile all defined functions which are later used
        for func in program.functions.iter().filter(|func| func.used) {
            self.compile_fn_definition(func);
        }

        // compile the main function
        self.compile_main_fn(&program.main_fn, main_fn);

        // verify the LLVM module when using debug
        self.module.verify().unwrap();

        // run optimizations on the IR
        self.optimize();

        // create the LLVM intermediate representation
        let llvm_ir = self.module.print_to_string().to_string();

        // build target-dependent object code
        let target = Target::from_triple(&self.target_triple)?;
        let Some(target_machine) = target.create_target_machine(
            &self.target_triple,
            "",
            "",
            self.optimization,
            RelocMode::PIC,
            CodeModel::Default,
        ) else { return Err(Error::NoTarget); };
        let objcode = target_machine.write_to_memory_buffer(&self.module, FileType::Object)?;

        Ok((objcode, llvm_ir))
    }

    fn optimize(&self) {
        let config = InitializationConfig::default();
        Target::initialize_all(&config);
        let pass_manager_builder = PassManagerBuilder::create();

        pass_manager_builder.set_optimization_level(self.optimization);
        pass_manager_builder.set_size_level(0); // either 0, 1, or 2

        let pass_manager = PassManager::create(());
        pass_manager_builder.populate_module_pass_manager(&pass_manager);
        pass_manager.run_on(&self.module);

        // perform LTO optimizations (for function inlining)
        let link_time_optimizations = PassManager::create(());
        pass_manager_builder.populate_lto_pass_manager(&link_time_optimizations, false, true);
        link_time_optimizations.run_on(&self.module);
    }

    fn compile_main_fn(&mut self, node: &'src AnalyzedBlock, main_fn: bool) {
        // main fn takes no arguments but returns an i8 (exit-code)
        let fn_type = self.context.i32_type().fn_type(&[], false);
        let main_fn = self.module.add_function(
            if main_fn { "main" } else { "_start" },
            fn_type,
            Some(Linkage::External),
        );

        // create basic block for the main function
        let main_basic_block = self.context.append_basic_block(main_fn, "entry");
        self.builder.position_at_end(main_basic_block);

        // set the current function to `main`
        self.curr_fn = Some(Function {
            name: "main",
            llvm_value: main_fn,
        });

        // create a new scope for the main function
        self.scopes.push(HashMap::new());

        // compile the function's body
        self.compile_block(node);

        // drop the function's scope
        self.scopes.pop();

        // return exit-code 0
        self.build_return(None);
    }

    /// Defines a new global variable with the given name and initializes it using the expression
    fn declare_global(&mut self, ident: &'src str, expression: &'src AnalyzedExpression) {
        let init_value = self.compile_expression(expression);
        let global = self.module.add_global(init_value.get_type(), None, ident);
        global.set_initializer(&init_value);
        // store the global variable in the globals vec
        self.globals.insert(ident, global.as_pointer_value());
    }

    fn compile_fn_signature(&mut self, node: &AnalyzedFunctionDefinition) {
        // create the function's parameters
        let params: Vec<BasicMetadataTypeEnum> = node
            .params
            .iter()
            .map(|param| match param.type_ {
                Type::Int => BasicMetadataTypeEnum::IntType(self.context.i64_type()),
                Type::Float => BasicMetadataTypeEnum::FloatType(self.context.f64_type()),
                Type::Bool => BasicMetadataTypeEnum::IntType(self.context.bool_type()),
                Type::Char => BasicMetadataTypeEnum::IntType(self.context.i8_type()),
                Type::Unit => BasicMetadataTypeEnum::IntType(self.context.bool_type()),
                Type::Never | Type::Unknown => {
                    unreachable!("the analyzer disallows these types to be used as parameters")
                }
            })
            .collect();

        // create the function's signature
        let signature = match node.return_type {
            Type::Int => self.context.i64_type().fn_type(&params, false),
            Type::Float => self.context.f64_type().fn_type(&params, false),
            Type::Unit => self.context.bool_type().fn_type(&params, false),
            Type::Char => self.context.i8_type().fn_type(&params, false),
            Type::Bool => self.context.bool_type().fn_type(&params, false),
            Type::Unknown | Type::Never => {
                unreachable!("functions do not return values of these types")
            }
        };

        // add the function to the LLVM module
        self.module
            .add_function(node.name, signature, Some(Linkage::External));
    }

    /// Defines a new function in the module and compiles it's body.
    /// Also allocates space for any function arguments later passed to the function.
    fn compile_fn_definition(&mut self, node: &'src AnalyzedFunctionDefinition) {
        let function = self
            .module
            .get_function(node.name)
            .expect("every fn exists");

        // create basic block for the function
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        // set the current function environment
        self.curr_fn = Some(Function {
            name: node.name,
            llvm_value: function,
        });

        // create new scope for the function
        self.scopes.push(HashMap::new());

        // bind each parameter to the original value (for later reference)
        for (i, param) in node.params.iter().enumerate() {
            match param.mutable {
                true => {
                    // allocate a pointer for each parameter (allows mutability)
                    let ptr = self.builder.build_alloca(
                        match param.type_ {
                            Type::Int => self.context.i64_type().as_basic_type_enum(),
                            Type::Float => self.context.f64_type().as_basic_type_enum(),
                            Type::Char => self.context.i8_type().as_basic_type_enum(),
                            Type::Bool => self.context.bool_type().as_basic_type_enum(),
                            Type::Unit => self.context.i8_type().as_basic_type_enum(),
                            Type::Never | Type::Unknown => {
                                unreachable!("such function params cannot exist")
                            }
                        },
                        param.name,
                    );
                    // get the param's value from the function
                    let value = function
                        .get_nth_param(i as u32)
                        .expect("this parameter exists");

                    // store the param value in the pointer
                    self.builder.build_store(ptr, value);
                    // insert the pointer / parameter into the current scope
                    self.scope_mut().insert(param.name, Variable::Mut(ptr));
                }
                false => {
                    // get the param's value from the function
                    let value = function
                        .get_nth_param(i as u32)
                        .expect("this parameter exists");
                    // insert the pointer / parameter into the current scope
                    self.scope_mut().insert(param.name, Variable::Const(value));
                }
            }
        }

        // build the result value of the function's body
        let return_value = self.compile_block(&node.block);

        // drop the function's scope
        self.scopes.pop();

        self.build_return(Some(return_value));
    }

    /// Compiles a block and returns its result value
    fn compile_block(&mut self, node: &'src AnalyzedBlock) -> BasicValueEnum<'ctx> {
        for stmt in &node.stmts {
            self.compile_statement(stmt);
        }
        // if there is an expression, return its value instead of `()`
        node.expr
            .as_ref()
            .map_or(self.unit_value(), |expr| self.compile_expression(expr))
    }

    /// Compiles a [`AnalyzedStatement`] without returning a value (statement: `()`).
    fn compile_statement(&mut self, node: &'src AnalyzedStatement) {
        match node {
            AnalyzedStatement::Let(node) => self.compile_let_statement(node),
            AnalyzedStatement::Return(node) => self.compile_return_statement(node),
            AnalyzedStatement::Loop(node) => self.compile_loop_statement(node),
            AnalyzedStatement::While(node) => self.compile_while_statement(node),
            AnalyzedStatement::For(node) => self.compile_for_statement(node),
            AnalyzedStatement::Break => self.compile_break_statement(),
            AnalyzedStatement::Continue => self.compile_continue_statement(),
            AnalyzedStatement::Expr(node) => {
                self.compile_expression(node);
            }
        }
    }

    /// Compiles a [`AnalyzedLetStmt`].
    /// Allocates a pointer for the value and stores the rhs value in it.
    /// Also inserts the pointer into the functions's [`HashMap`] for later use.
    fn compile_let_statement(&mut self, node: &'src AnalyzedLetStmt) {
        let rhs = self.compile_expression(&node.expr);

        // if the variable is mutable, no pointer allocations are required
        match node.mutable {
            true => {
                // allocate a pointer for the value
                let ptr = self.alloc_ptr(node.name, node.expr.result_type(), rhs.get_type());

                // store the rhs value in the pointer
                self.builder.build_store(ptr, rhs);

                // insert the pointer into the current scope (for later reference)
                self.scope_mut().insert(node.name, Variable::Mut(ptr));
            }
            false => {
                self.scope_mut().insert(node.name, Variable::Const(rhs));
            }
        };
    }

    fn alloc_ptr(
        &mut self,
        name: &str,
        type_: Type,
        llvm_type: BasicTypeEnum<'ctx>,
    ) -> PointerValue<'ctx> {
        // if there is a (root) loop, use its allocations
        match self.loops.first_mut() {
            Some(curr_loop) => {
                let mut ptr = None;
                for (idx, (alloc_ident, alloc_type, alloc_ptr)) in
                    curr_loop.allocations.iter_mut().enumerate()
                {
                    if *alloc_ident == name && *alloc_type == type_ {
                        ptr = Some(*alloc_ptr);
                        // remove this allocation from the vec
                        curr_loop.allocations.remove(idx);
                        break;
                    }
                }
                ptr.expect("every loop allocation exists")
            }
            None => {
                // if there is no loop with allocations, allocate a pointer
                self.builder.build_alloca(llvm_type, name)
            }
        }
    }

    /// Compiles a return statement with an optional expression as it's value.
    /// If there is no optional expression, `()` / void is used as the return type.
    fn compile_return_statement(&mut self, node: &'src AnalyzedReturnStmt) {
        match node {
            Some(expr) => {
                let value = self.compile_expression(expr);
                self.build_return(Some(value))
            }
            None => self.build_return(None),
        };
    }

    /// Transforms a vector containing allocations tracked by the analyzer into a vector conainting
    /// allocated pointers. For each element, a new LLVM pointer allocation is made.
    /// Used in loop statements in order to perform allocations upfront.
    fn do_loop_allocations(
        &mut self,
        src: &[(&'src str, Type)],
    ) -> Vec<(String, Type, PointerValue<'ctx>)> {
        src.iter()
            .map(|(ident, type_)| {
                let ptr = self.builder.build_alloca(self.to_llvm_type(*type_), ident);
                self.scope_mut().insert(ident, Variable::Mut(ptr));
                (ident.to_string(), *type_, ptr)
            })
            .collect()
    }

    /// Compiles an [`AnalyzedLoopStmt`].
    /// Generates a start basic block and a after basic block.
    /// When break / continue is used in the loop, it jumps to the start or end blocks.
    fn compile_loop_statement(&mut self, node: &'src AnalyzedLoopStmt) {
        // create the loop_head block
        let loop_head = self
            .context
            .append_basic_block(self.curr_fn().llvm_value, "loop_head");

        // create the after_loop block
        let after_loop = self
            .context
            .append_basic_block(self.curr_fn().llvm_value, "after_loop");

        // push a new scope for the loop
        self.scopes.push(HashMap::new());

        // set the loop metadata so that the inner block can use it
        let allocations = self.do_loop_allocations(&node.allocations);
        self.loops.push(Loop {
            loop_head,
            after_loop,
            allocations,
        });

        // enter the loop from outside
        self.builder.build_unconditional_branch(loop_head);

        // compile the loop body
        self.builder.position_at_end(loop_head);
        self.compile_block(&node.block);

        // jump back to the loop head
        self.builder.build_unconditional_branch(loop_head);

        // drop the scope
        self.scopes.pop();

        // remove the loop from `loops`
        self.loops.pop();

        // place the builder cursor at the end of the `after_loop`
        self.builder.position_at_end(after_loop);
    }

    /// Compiles an [`AnalyzedWhileStmt`].
    /// Checks the provided condition before attempting a new iteration.
    fn compile_while_statement(&mut self, node: &'src AnalyzedWhileStmt) {
        // create the `while_head` block
        let while_head = self
            .context
            .append_basic_block(self.curr_fn().llvm_value, "while_head");

        // create the `while_body` block
        let while_body = self
            .context
            .append_basic_block(self.curr_fn().llvm_value, "while_body");

        // create the `after_while` block
        let after_while = self
            .context
            .append_basic_block(self.curr_fn().llvm_value, "after_while");

        // push a new scope for the loop
        self.scopes.push(HashMap::new());

        // set the loop metadata so that the inner block can use it
        let allocations = self.do_loop_allocations(&node.allocations);
        self.loops.push(Loop {
            loop_head: while_head,
            after_loop: after_while,
            allocations,
        });

        // enter the loop from outside
        self.builder.build_unconditional_branch(while_head);

        // compile the condition check
        self.builder.position_at_end(while_head);
        let cond = self.compile_expression(&node.cond);
        // if the condition is true, jump into the while body, otherwise, quit the loop
        self.builder
            .build_conditional_branch(cond.into_int_value(), while_body, after_while);

        // compile the loop body
        self.builder.position_at_end(while_body);
        self.compile_block(&node.block);

        // jump back to the loop head
        self.builder.build_unconditional_branch(while_head);

        // drop the loop scope
        self.scopes.pop();

        // remove the loop from `loops`
        self.loops.pop();

        // place the builder cursor at the end of the `after_loop`
        self.builder.position_at_end(after_while);
    }

    /// Compiles an [`AnalyzedForStmt`].
    /// The init expression is compiled at the beginning.
    /// Each iteration will only take place if the condition expression evaluates to `true`.
    /// At the beginning of each iteration, the update expression is invoked.
    fn compile_for_statement(&mut self, node: &'src AnalyzedForStmt) {
        // create the `for_head` block
        let for_head = self
            .context
            .append_basic_block(self.curr_fn().llvm_value, "for_head");

        // create the `for_body` block
        let for_body = self
            .context
            .append_basic_block(self.curr_fn().llvm_value, "for_body");

        // create the `for_update` block
        let for_update = self
            .context
            .append_basic_block(self.curr_fn().llvm_value, "for_update");

        // create the `after_for` block
        let after_for = self
            .context
            .append_basic_block(self.curr_fn().llvm_value, "after_for");

        // insert the induction variable into the current scope
        let induction_var = self.compile_expression(&node.initializer);

        // allocate a pointer for the induction variable
        let ptr = self.alloc_ptr(
            node.ident,
            node.initializer.result_type(),
            induction_var.get_type(),
        );

        // store the value in the pointer
        self.builder.build_store(ptr, induction_var);

        // push a new scope for the loop
        self.scopes.push(HashMap::new());

        // set the loop metadata so that the inner block can use it
        let allocations = self.do_loop_allocations(&node.allocations);
        self.loops.push(Loop {
            loop_head: for_head,
            after_loop: after_for,
            allocations,
        });

        // add the pointer to the loop's scope
        self.scope_mut().insert(node.ident, Variable::Mut(ptr));

        // enter the loop from outside
        self.builder.build_unconditional_branch(for_head);

        // compile the condition check
        self.builder.position_at_end(for_head);
        let cond = self.compile_expression(&node.cond);
        // if the condition is true, jump into the for body, otherwise, quit the loop
        self.builder
            .build_conditional_branch(cond.into_int_value(), for_body, after_for);

        // compile the loop body
        self.builder.position_at_end(for_body);
        self.compile_block(&node.block);
        // jump to the update block
        self.builder.build_unconditional_branch(for_update);

        // compile the update expression
        self.builder.position_at_end(for_update);
        self.compile_expression(&node.update);
        // jump back to the loop head
        self.builder.build_unconditional_branch(for_head);

        // drop the scope
        self.scopes.pop();

        // remove the loop from `loops`
        self.loops.pop();

        // place the builder cursor at the end of the `after_loop`
        self.builder.position_at_end(after_for);
    }

    /// Compiles a break statement.
    /// The task of the break statement is to jump to the basic block after the loop.
    /// This basic block is saved under the `curr_loop` field of the compiler.
    fn compile_break_statement(&mut self) {
        let after_loop_block = self
            .loops
            .last()
            .as_ref()
            .expect("break is only called in loop bodies")
            .after_loop;
        self.builder.build_unconditional_branch(after_loop_block);
    }

    /// Compiles a continue statement.
    /// The task of the continue statement is to jump to the basic block at the start of the loop.
    /// This basic block is saved under the `curr_loop` field of the compiler.
    fn compile_continue_statement(&mut self) {
        let loop_start_block = self
            .loops
            .last()
            .as_ref()
            .expect("continue is only called in loop bodies")
            .loop_head;
        self.builder.build_unconditional_branch(loop_start_block);
    }

    /// Compiles an [`AnalyzedExpression`].
    /// Creates constant values for simple atoms, such as int, float, char, and bool.
    /// Otherwise, the function invokes other expressions.
    fn compile_expression(&mut self, node: &'src AnalyzedExpression) -> BasicValueEnum<'ctx> {
        match node {
            AnalyzedExpression::Int(value) => self
                .context
                .i64_type()
                .const_int(*value as u64, false)
                .as_basic_value_enum(),

            AnalyzedExpression::Float(value) => self
                .context
                .f64_type()
                .const_float(*value)
                .as_basic_value_enum(),
            AnalyzedExpression::Char(value) => self
                .context
                .i8_type()
                .const_int(*value as u64, false)
                .as_basic_value_enum(),
            AnalyzedExpression::Bool(value) => {
                // create an i8 which is either 0 (false) or 1 (true)
                let bool_int = self.context.i8_type().const_int(u64::from(*value), false);
                // convert the i8 to a LLVM boolean
                self.builder
                    .build_int_cast(bool_int, self.context.bool_type(), "bool")
                    .as_basic_value_enum()
            }
            AnalyzedExpression::Ident(name) => {
                let variable = self.resolve_name(name.ident);
                match variable {
                    Variable::Const(value) => value,
                    Variable::Mut(ptr) => self.builder.build_load(ptr, name.ident),
                }
            }
            AnalyzedExpression::Call(node) => self.compile_call_expression(node),
            AnalyzedExpression::Grouped(node) => self.compile_expression(node),
            AnalyzedExpression::Block(node) => self.compile_block(node),
            AnalyzedExpression::Infix(node) => self.compile_infix_expression(node),
            AnalyzedExpression::Prefix(node) => self.compile_prefix_expression(node),
            AnalyzedExpression::Cast(node) => self.compile_cast_expression(node),
            AnalyzedExpression::Assign(node) => {
                self.compile_assign_expression(node);
                // the result type of an assignment is `()`
                self.unit_value()
            }
            AnalyzedExpression::If(node) => self.compile_if_expression(node),
        }
    }

    /// Helper function for resolving identifier names.
    /// Searches the scopes first. If no match was found, the fitting global variable is returned.
    fn resolve_name(&self, name: &str) -> Variable<'ctx> {
        for scope in self.scopes.iter().rev() {
            if let Some(&variable) = scope.get(name) {
                return variable;
            }
        }
        match self.module.get_global(name) {
            Some(global) => Variable::Mut(global.as_pointer_value()),
            None => unreachable!("every name used is either a var or global: {name}"),
        }
    }

    /// Compiles an [`AnalyzedCallExpr`] and returns the result of the call.
    /// If a builtin is called, it is declared just-in-time to avoid redundant declarations.
    fn compile_call_expression(&mut self, node: &'src AnalyzedCallExpr) -> BasicValueEnum<'ctx> {
        // handle any builtin functions
        let func = match node.func {
            "exit" => {
                // the exit function requires 1 i64 and returns void
                let exit_type = self.context.void_type().fn_type(
                    &[BasicMetadataTypeEnum::IntType(self.context.i64_type())],
                    false,
                );
                // either get the function from the module or declare it just-in-time
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
            "main" => self
                .module
                .get_function("_start")
                .expect("main function exists"),
            // for user-defined funcs: look up the identifier in the module
            _ => self
                .module
                .get_function(node.func)
                .expect("this function exists"),
        };

        // create the function's arguments
        let args: Vec<BasicMetadataValueEnum> = node
            .args
            .iter()
            .map(|arg| BasicMetadataValueEnum::from(self.compile_expression(arg)))
            .collect();

        // perform the function call
        let res = self
            .builder
            .build_call(func, &args, format!("ret_{}", node.func).as_str())
            .try_as_basic_value();

        res.left_or(self.unit_value())
    }

    /// Helper function for performing infix operations on values.
    fn infix_helper(
        &mut self,
        lhs_type: Type,
        op: InfixOp,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        match lhs_type {
            Type::Float => {
                let lhs = lhs.into_float_value();
                let rhs = rhs.into_float_value();
                match op {
                    InfixOp::Plus => self.builder.build_float_add(lhs, rhs, "f_sum"),
                    InfixOp::Minus => self.builder.build_float_sub(lhs, rhs, "f_sum"),
                    InfixOp::Mul => self.builder.build_float_mul(lhs, rhs, "f_prod"),
                    InfixOp::Div => self.builder.build_float_div(lhs, rhs, "f_prod"),
                    InfixOp::Rem => self.builder.build_float_rem(lhs, rhs, "f_rem"),
                    // comparison operators (result in bool)
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
            // even if some combinations are invalid, this is OK because the analyzer prevents
            // illegal cases.
            Type::Int | Type::Bool => {
                let lhs = lhs.into_int_value();
                let rhs = rhs.into_int_value();
                match op {
                    InfixOp::Plus => self.builder.build_int_add(lhs, rhs, "i_sum"),
                    InfixOp::Minus => self.builder.build_int_sub(lhs, rhs, "i_sum"),
                    InfixOp::Mul => self.builder.build_int_mul(lhs, rhs, "i_prod"),
                    InfixOp::Div => self.builder.build_int_signed_div(lhs, rhs, "i_prod"),
                    InfixOp::Rem => self.builder.build_int_signed_rem(lhs, rhs, "i_rem"),
                    InfixOp::Pow => self.__rush_internal_pow(lhs, rhs),
                    InfixOp::Shl => self.builder.build_left_shift(lhs, rhs, "i_shl"),
                    InfixOp::Shr => self.builder.build_right_shift(lhs, rhs, true, "i_shr"),
                    InfixOp::BitOr => self.builder.build_or(lhs, rhs, "i_bor"),
                    InfixOp::BitAnd => self.builder.build_and(lhs, rhs, "i_band"),
                    InfixOp::BitXor => self.builder.build_xor(lhs, rhs, "i_bxor"),
                    // comparison operators (result in bool)
                    op => {
                        let (op, label) = match op {
                            InfixOp::Eq => (IntPredicate::EQ, "i_eq"),
                            InfixOp::Neq => (IntPredicate::NE, "i_neq"),
                            InfixOp::Gt => (IntPredicate::SGT, "i_gt"),
                            InfixOp::Gte => (IntPredicate::SGE, "i_gte"),
                            InfixOp::Lt => (IntPredicate::SLT, "i_lt"),
                            InfixOp::Lte => (IntPredicate::SLE, "i_lte"),
                            _ => unreachable!("other operators cannot be used on int: {op:?}"),
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
                match op {
                    InfixOp::Plus => {
                        let res = self.builder.build_int_add(lhs, rhs, "i_sum");
                        self.builder.build_and(
                            res,
                            self.context.i8_type().const_int(127, false),
                            "c_mask",
                        )
                    }
                    InfixOp::Minus => {
                        let res = self.builder.build_int_sub(lhs, rhs, "i_sum");
                        self.builder.build_and(
                            res,
                            self.context.i8_type().const_int(127, false),
                            "c_mask",
                        )
                    }
                    // comparison operators (result in bool)
                    op => {
                        let (op, label) = match op {
                            InfixOp::Eq => (IntPredicate::EQ, "i_eq"),
                            InfixOp::Neq => (IntPredicate::NE, "i_neq"),
                            InfixOp::Gt => (IntPredicate::SGT, "i_gt"),
                            InfixOp::Gte => (IntPredicate::SGE, "i_gte"),
                            InfixOp::Lt => (IntPredicate::SLT, "i_lt"),
                            InfixOp::Lte => (IntPredicate::SLE, "i_lte"),
                            _ => unreachable!("other operators cannot be used on int: {op:?}"),
                        };
                        return self
                            .builder
                            .build_int_compare(op, lhs, rhs, label)
                            .as_basic_value_enum();
                    }
                }
                .as_basic_value_enum()
            }
            Type::Unknown | Type::Unit | Type::Never => {
                unreachable!("these types cannot be used in an infix expression")
            }
        }
    }

    fn infix_logical_branch(
        &mut self,
        node: &'src AnalyzedExpression,
        merge_block: BasicBlock<'ctx>,
    ) -> (BasicValueEnum<'ctx>, BasicBlock<'ctx>) {
        let rhs_value = self.compile_expression(node);

        let branch_block = self.curr_block();
        self.builder.build_unconditional_branch(merge_block);
        (rhs_value, branch_block)
    }

    /// Compiles an [`AnalyzedInfixExpr`].
    /// Handles the `bool || bool` and `bool && bool` edge cases directly.
    /// Invokes the `infix_helper` function for any other types or operations.
    fn compile_infix_expression(&mut self, node: &'src AnalyzedInfixExpr) -> BasicValueEnum<'ctx> {
        match (node.lhs.result_type(), node.op) {
            // uses an if-else in order to skip evaluation of the rhs if the lhs is `true`
            (Type::Bool, InfixOp::Or) => {
                // compile the condition (lhs)
                let lhs_cond = self.compile_expression(&node.lhs);

                // create the basic blocks
                let lhs_true_block = self
                    .context
                    .append_basic_block(self.curr_fn().llvm_value, "lhs_is_true");
                let lhs_false_block = self
                    .context
                    .append_basic_block(self.curr_fn().llvm_value, "lhs_is_false");
                let merge_block = self
                    .context
                    .append_basic_block(self.curr_fn().llvm_value, "logical_or_merge");

                self.builder.build_conditional_branch(
                    lhs_cond.into_int_value(),
                    lhs_true_block,
                    lhs_false_block,
                );

                // if the lhs is true, stop here and return true
                self.builder.position_at_end(lhs_true_block);
                let lhs_true_value = self.context.bool_type().const_int(1, false);
                self.builder.build_unconditional_branch(merge_block);

                // if the value is false, execute the rhs and return its value
                self.builder.position_at_end(lhs_false_block);
                //let rhs_value = self.compile_expression(&node.rhs);
                let (rhs_value, rhs_block) = self.infix_logical_branch(&node.rhs, merge_block);

                // insert a phi node to pick the correct value
                self.builder.position_at_end(merge_block);

                let phi = self
                    .builder
                    .build_phi(self.context.bool_type(), "logical_or_res");
                phi.add_incoming(&[(&lhs_true_value, lhs_true_block), (&rhs_value, rhs_block)]);
                // return the value of the phi
                phi.as_basic_value()
            }
            // uses an if-else in order to skip evaluation of the rhs if the lhs is `false`
            (Type::Bool, InfixOp::And) => {
                // compile the condition (lhs)
                let lhs_cond = self.compile_expression(&node.lhs);

                // create the basic blocks
                let lhs_true_block = self
                    .context
                    .append_basic_block(self.curr_fn().llvm_value, "lhs_is_true");
                let lhs_false_block = self
                    .context
                    .append_basic_block(self.curr_fn().llvm_value, "lhs_is_false");
                let merge_block = self
                    .context
                    .append_basic_block(self.curr_fn().llvm_value, "logical_or_merge");

                self.builder.build_conditional_branch(
                    lhs_cond.into_int_value(),
                    lhs_true_block,
                    lhs_false_block,
                );

                // if the lhs is true, execute the rhs and return its value
                self.builder.position_at_end(lhs_true_block);
                let (rhs_value, rhs_block) = self.infix_logical_branch(&node.rhs, merge_block);

                // if the lhs value is false, stop here and return valse
                self.builder.position_at_end(lhs_false_block);
                let false_value = self.context.bool_type().const_zero();
                self.builder.build_unconditional_branch(merge_block);

                // insert a phi node to pick the correct value
                self.builder.position_at_end(merge_block);

                let phi = self
                    .builder
                    .build_phi(self.context.bool_type(), "logical_and_res");
                phi.add_incoming(&[(&rhs_value, rhs_block), (&false_value, lhs_false_block)]);

                // return the value of the phi
                phi.as_basic_value()
            }
            // invoke the infix helper for any other types
            (type_, _) => {
                let lhs = self.compile_expression(&node.lhs);
                let rhs = self.compile_expression(&node.rhs);
                self.infix_helper(type_, node.op, lhs, rhs)
            }
        }
    }

    fn compile_prefix_expression(
        &mut self,
        node: &'src AnalyzedPrefixExpr,
    ) -> BasicValueEnum<'ctx> {
        let base = self.compile_expression(&node.expr);

        match (node.expr.result_type(), node.op) {
            (Type::Int, PrefixOp::Neg) => self
                .builder
                .build_int_neg(base.into_int_value(), "neg")
                .as_basic_value_enum(),
            (Type::Int, PrefixOp::Not) => self
                .builder
                .build_not(base.into_int_value(), "not")
                .as_basic_value_enum(),
            (Type::Float, PrefixOp::Neg) => self
                .builder
                .build_float_neg(base.into_float_value(), "neg")
                .as_basic_value_enum(),
            (Type::Bool, PrefixOp::Not) => self
                .builder
                .build_int_compare(
                    IntPredicate::EQ,
                    base.into_int_value(),
                    self.context.bool_type().const_zero(),
                    "bool_neg",
                )
                .as_basic_value_enum(),
            _ => unreachable!("other types are not supported by prefix"),
        }
    }

    /// Compiles an [`AnalyzedCastExpr`], such as `42 as char` and returns the resulting value.
    fn compile_cast_expression(&mut self, node: &'src AnalyzedCastExpr) -> BasicValueEnum<'ctx> {
        let lhs = self.compile_expression(&node.expr);

        match (node.expr.result_type(), node.type_) {
            // if the lhs == rhs, no operations is to be done
            (l, typ) if l == typ => lhs,
            (Type::Int, Type::Float) => {
                let lhs_int = lhs.into_int_value();
                self.builder
                    .build_signed_int_to_float(lhs_int, self.context.f64_type(), "if_cast")
                    .as_basic_value_enum()
            }
            // converting a type to a char requires additional bounds checks
            // because valid ASCII chars lie in the range 0 - 127, these checks must be done.
            // the cast operation therefore invokes a builtin helper function.
            (Type::Int, Type::Char) => self.__rush_internal_cast_int_to_char(lhs.into_int_value()),
            (Type::Char, Type::Int) => self
                .builder
                .build_int_cast(lhs.into_int_value(), self.context.i64_type(), "ci_cast")
                .as_basic_value_enum(),
            (Type::Int, Type::Bool) => {
                let lhs_int = lhs.into_int_value();
                self.builder
                    .build_int_compare(
                        IntPredicate::NE,
                        lhs_int,
                        self.context.i64_type().const_zero(),
                        "ib_cast_cmp",
                    )
                    .as_basic_value_enum()
            }
            (Type::Float, Type::Int) => {
                let lhs_bool = lhs.into_float_value();
                self.builder
                    .build_float_to_signed_int(lhs_bool, self.context.i64_type(), "fi_cast")
                    .as_basic_value_enum()
            }
            // converting a type to a char requires additional bounds checks
            // because valid ASCII chars lie in the range 0 - 127, these checks must be done.
            // the cast operation therefore invokes a builtin helper function.
            (Type::Float, Type::Char) => {
                self.__rush_internal_cast_float_to_char(lhs.into_float_value())
            }
            (Type::Float, Type::Bool) => self
                .builder
                .build_float_compare(
                    FloatPredicate::ONE,
                    lhs.into_float_value(),
                    self.context.f64_type().const_zero(),
                    "fb_cast",
                )
                .as_basic_value_enum(),
            (Type::Bool, Type::Int) => self
                .builder
                .build_int_cast_sign_flag(
                    lhs.into_int_value(),
                    self.context.i64_type(),
                    false,
                    "bi_cast",
                )
                .as_basic_value_enum(),
            (Type::Bool, Type::Float) => self
                .builder
                .build_unsigned_int_to_float(
                    lhs.into_int_value(),
                    self.context.f64_type(),
                    "bf_cast",
                )
                .as_basic_value_enum(),
            (Type::Bool, Type::Char) => self
                .builder
                .build_int_cast_sign_flag(
                    lhs.into_int_value(),
                    self.context.i8_type(),
                    false,
                    "bc_cast",
                )
                .as_basic_value_enum(),
            (Type::Char, Type::Float) => self
                .builder
                .build_unsigned_int_to_float(
                    lhs.into_int_value(),
                    self.context.f64_type(),
                    "cf_cast",
                )
                .as_basic_value_enum(),
            (Type::Char, Type::Bool) => self
                .builder
                .build_int_cast(lhs.into_int_value(), self.context.bool_type(), "cb_cast")
                .as_basic_value_enum(),
            _ => unreachable!("other casts are impossible"),
        }
    }

    /// Compiles an [`AnalyzedAssignExpr`] by performing its operation and assignment.
    fn compile_assign_expression(&mut self, node: &'src AnalyzedAssignExpr) {
        // get the pointer of the assignee
        let ptr = match self.resolve_name(node.assignee) {
            Variable::Mut(ptr) => ptr,
            Variable::Const(_) => unreachable!("can only assign to mutable variables"),
        };

        match (node.op, node.expr.result_type()) {
            (AssignOp::Basic, _) => {
                let rhs = self.compile_expression(&node.expr);
                // store the rhs value in the pointer
                self.builder.build_store(ptr, rhs);
            }
            (op, Type::Int | Type::Float | Type::Bool) => {
                // compile the value of the rhs for later use
                let rhs = self.compile_expression(&node.expr);
                // load the value from the pointer
                let assignee = self.builder.build_load(ptr, node.assignee);
                // perform the assign op operation on the pointer value and the rhs
                let res = self.infix_helper(Type::Int, InfixOp::from(op), assignee, rhs);
                // store the resulting value in the pointer
                self.builder.build_store(ptr, res);
            }
            _ => unreachable!("other types cannot be used in this context"),
        }
    }

    /// Compiles an [`AnalyzedIfExpr`] by inserting a branch construct.
    fn compile_if_expression(&mut self, node: &'src AnalyzedIfExpr) -> BasicValueEnum<'ctx> {
        // compile the if condition
        let cond = self.compile_expression(&node.cond);

        // create basic blocks for the `then` and `else` branches
        let then_block = self
            .context
            .append_basic_block(self.curr_fn().llvm_value, "then");
        let merge_block = self
            .context
            .append_basic_block(self.curr_fn().llvm_value, "merge");

        // create an else block if specified in the AST
        if let Some(else_node) = &node.else_block {
            let else_block = self
                .context
                .append_basic_block(self.curr_fn().llvm_value, "else");

            // branch between the `then` and `else` blocks using the condition
            self.builder
                .build_conditional_branch(cond.into_int_value(), then_block, else_block);

            // compile the `then` branch
            self.builder.position_at_end(then_block);
            let (_, then_option) = self.compile_branch(&node.then_block, merge_block);

            // compile the `else` branch
            self.builder.position_at_end(else_block);
            let (_, else_option) = self.compile_branch(else_node, merge_block);

            // place the builder at the end of the `merge` block (exit block)
            self.builder.position_at_end(merge_block);

            // inserts a phi node in order to use the value of the branch which was taken.
            // if a branch has terminated early (return), no phi node should be inserted
            match (then_option, else_option) {
                (Some((then_value, then_branch)), Some((else_value, else_branch))) => {
                    let phi = self.builder.build_phi(then_value.get_type(), "if_res");
                    phi.add_incoming(&[(&then_value, then_branch), (&else_value, else_branch)]);
                    phi.as_basic_value()
                }
                (Some((then_value, _)), None) => then_value,
                (None, Some((else_value, _))) => else_value,
                (None, None) => {
                    // in this case, the block is unreachable due to a previous return
                    // the compiler still needs a value so a `unit` value is returned
                    self.builder.build_unreachable();
                    self.unit_value()
                }
            }
        } else {
            // if there is no `else` block, just branch between the `if` and `merge` blocks
            self.builder
                .build_conditional_branch(cond.into_int_value(), then_block, merge_block);

            // compile the `then` branch
            self.builder.position_at_end(then_block);
            self.compile_branch(&node.then_block, merge_block);

            // the merge block just returns a unit value
            // if without else is only possible if the result of the if-expr is `()`
            self.builder.position_at_end(merge_block);
            self.unit_value()
        }
    }

    /// Helper function which returns a zero i1 as a unit-value
    fn unit_value(&self) -> BasicValueEnum<'ctx> {
        let i1 = self.context.bool_type();
        i1.const_zero().into()
    }

    /// Compiles a branch in an [`AnalyzedIfExpr`].
    /// Automatically jumps to the correct merge block when done.
    /// Handles the edge case when the block uses `return`
    /// Automatically pushes and pops the branch's scopes
    fn compile_branch(
        &mut self,
        node: &'src AnalyzedBlock,
        end_block: BasicBlock<'ctx>,
    ) -> (
        BasicTypeEnum<'ctx>,
        Option<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)>,
    ) {
        // push a new scope for the branch
        self.scopes.push(HashMap::new());

        // compile the block
        let branch_value = self.compile_block(node);

        // pop the branch scope
        self.scopes.pop();

        // if the block was terminated prior, do not return the branch block
        if self.current_instruction_is_block_terminator() {
            (branch_value.get_type(), None)
        } else {
            let branch_block = self.curr_block();
            self.builder.build_unconditional_branch(end_block);
            (branch_value.get_type(), Some((branch_value, branch_block)))
        }
    }

    /// Helper function for accessing the current LLVM basic block
    fn curr_block(&self) -> BasicBlock<'ctx> {
        self.builder
            .get_insert_block()
            .expect("this function is only used if a block was previously inserted")
    }

    /// Checks if the current LLVM builder instruction is used to terminate a block
    fn current_instruction_is_block_terminator(&self) -> bool {
        let instruction = self.curr_block().get_last_instruction();
        matches!(
            instruction.map(|instruction| instruction.get_opcode()),
            Some(
                InstructionOpcode::Return | InstructionOpcode::Unreachable | InstructionOpcode::Br
            )
        )
    }

    /// Adds a return instruction using the specified value.
    /// However, if the current basic block already contains a block terminator (e.g. return / unreachable),
    /// the insertion is omitted in order to prevent an LLVM error
    fn build_return(&mut self, return_value: Option<BasicValueEnum<'ctx>>) {
        if !self.current_instruction_is_block_terminator() {
            match (return_value, self.curr_fn().name) {
                (_, "main") => {
                    let success = self.context.i32_type().const_zero();
                    self.builder.build_return(Some(&success))
                }
                (Some(value), _) => self.builder.build_return(Some(&value)),
                (None, _) => self.builder.build_return(None),
            };
        }
    }

    fn to_llvm_type(&self, type_: Type) -> BasicTypeEnum<'ctx> {
        match type_ {
            Type::Int => self.context.i64_type().as_basic_type_enum(),
            Type::Bool => self.context.bool_type().as_basic_type_enum(),
            Type::Float => self.context.f64_type().as_basic_type_enum(),
            Type::Char => self.context.i8_type().as_basic_type_enum(),
            Type::Unit => self.context.bool_type().as_basic_type_enum(),
            Type::Never | Type::Unknown => {
                unreachable!("cannot convert these types")
            }
        }
    }
}

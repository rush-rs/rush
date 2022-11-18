#![allow(dead_code)] // TODO: remove this attribute

use std::{collections::HashMap, fmt::Display};

use rush_analyzer::{
    ast::{
        AnalyzedBlock, AnalyzedCallExpr, AnalyzedExpression, AnalyzedFunctionDefinition,
        AnalyzedLetStmt, AnalyzedProgram, AnalyzedStatement,
    },
    Type,
};

use crate::{
    instruction::{FldType, Instruction, Pointer},
    register::{FloatRegister, IntRegister, Register},
};

pub struct Compiler {
    /// Exported functions
    pub(crate) exports: Vec<String>,
    /// Sections / basic blocks which contain instructions.
    pub(crate) blocks: Vec<Block>,
    /// Points to the current section which is inserted to
    pub(crate) curr_block: usize,
    /// Data section for storing global variables.
    pub(crate) data_section: Vec<DataObj>,
    /// Read-only data section for storing constant values.
    pub(crate) rodata_section: Vec<DataObj>,
    /// Holds metadata about the current function
    pub(crate) curr_fn: Function,
    /// Saves the scopes. The last element is the most recent scope.
    pub(crate) scopes: Vec<HashMap<String, Option<i64> /* sp offset */>>,
    /// Specifies all registers which are currently in use and may not be overwritten.
    pub(crate) used_registers: Vec<IntRegister>,
    /// Specifies all float registers which are currently in use and may not be overwritten.
    pub(crate) used_float_registers: Vec<FloatRegister>,
}

pub(crate) struct Block {
    pub(crate) label: String,
    pub(crate) instructions: Vec<Instruction>,
}

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n{}:\n{}",
            self.label,
            self.instructions
                .iter()
                .map(|i| format!("    {i}\n"))
                .collect::<String>()
        )
    }
}

pub(crate) struct Function {
    /// Specifies how many bytes of stack space need to be allocated for the current function.
    /// Include the necessary allocation for `ra` or `fp`
    pub(crate) fp: usize,
}

pub(crate) struct DataObj {
    pub(crate) label: String,
    pub(crate) data: DataObjType,
}

impl Display for DataObj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:\n    {}", self.label, self.data)
    }
}

pub(crate) enum DataObjType {
    Float(f64),
    Dword(i64),
    Byte(i64),
}

impl Display for DataObjType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(inner) => write!(
                f,
                ".dword {:#018x}  # = {inner}{zero}",
                inner.to_bits(),
                zero = if inner.fract() == 0.0 { ".0" } else { "" }
            ),
            Self::Dword(inner) => write!(f, ".dword {inner:#018x}  # = {inner}"),
            Self::Byte(inner) => write!(f, ".byte {inner:#04x}"),
        }
    }
}

impl Compiler {
    fn position_at_end(&mut self, label: &str) {
        self.curr_block = self
            .blocks
            .iter()
            .position(|s| s.label == label)
            .expect("cannot insert at invalid label: {label}")
    }

    pub fn new() -> Self {
        Self {
            blocks: vec![],
            exports: vec![],
            curr_block: 0,
            data_section: vec![],
            rodata_section: vec![],
            scopes: vec![],
            curr_fn: Function { fp: 0 },
            used_registers: vec![],
            used_float_registers: vec![],
        }
    }

    pub fn compile(&mut self, ast: AnalyzedProgram) -> String {
        for _var in ast.globals {
            // TODO: add zero initializer, init variables in
        }

        self.declare_main_fn(ast.main_fn);

        for func in ast.functions.into_iter().filter(|f| f.used) {
            self.function_declaration(func)
        }

        let mut output = String::new();

        output += ".section .text\n";

        // string generation
        output += &self
            .exports
            .iter()
            .map(|e| format!(".global {e}\n"))
            .collect::<String>();

        // block generation
        output += &self
            .blocks
            .iter()
            .map(|d| d.to_string())
            .collect::<String>();

        // .data generation
        if !self.data_section.is_empty() {
            output += &format!(
                "\n.section .data\n{}",
                self.data_section
                    .iter()
                    .map(|d| format!("\n{d}\n"))
                    .collect::<String>()
            );
        }

        // .rodata generation
        if !self.rodata_section.is_empty() {
            output += &format!(
                "\n.section .rodata\n{}",
                self.rodata_section
                    .iter()
                    .map(|d| format!("\n{d}\n"))
                    .collect::<String>()
            );
        }

        output
    }

    fn declare_main_fn(&mut self, node: AnalyzedBlock) {
        // append block for the function
        let block_label = "_start";
        self.append_block(block_label.into());
        self.exports.push(block_label.into());
        self.position_at_end(block_label);

        self.function_body(node);

        // exit with code 0
        self.insert(Instruction::Li(IntRegister::A0, 0));
        self.insert(Instruction::Call("exit".into()));
    }

    fn declare_global(&mut self, _ident: String, _value: AnalyzedExpression) {
        todo!()
    }

    fn function_declaration(&mut self, node: AnalyzedFunctionDefinition) {
        // append block for the function
        let block_label = format!(".{}", node.name);
        self.append_block(block_label.clone());
        self.position_at_end(&block_label);

        // push a scope
        self.push_scope();
        self.function_body(node.block);
        self.pop_scope();
    }

    fn function_body(&mut self, node: AnalyzedBlock) {
        // compile each statement
        for stmt in node.stmts {
            self.statement(stmt);
        }

        // place the result of the optional expression in the return value register(s)
        if let Some(expr) = node.expr {
            let res_reg = self.expression(expr);
            match res_reg {
                Some(Register::Int(IntRegister::A0))
                | Some(Register::Float(FloatRegister::Fa0)) => {} // already in target register
                Some(Register::Int(reg)) => {
                    self.insert(Instruction::Mov(IntRegister::A0, reg));
                }
                Some(Register::Float(reg)) => {
                    self.insert(Instruction::Fmov(FloatRegister::Fa0, reg));
                }
                None => {} // do nothing with unit values
            }
        }
    }

    fn block(&mut self, node: AnalyzedBlock) -> Option<Register> {
        // push a new scope
        self.push_scope();

        for stmt in node.stmts {
            self.statement(stmt)
        }

        // pop the scope again
        self.pop_scope();

        // return expression register if there is an expr
        match node.expr {
            Some(expr) => self.expression(expr),
            None => None,
        }
    }

    fn statement(&mut self, node: AnalyzedStatement) {
        match node {
            AnalyzedStatement::Let(node) => self.let_statement(node),
            AnalyzedStatement::Return(_) => todo!(),
            AnalyzedStatement::Loop(_) => todo!(),
            AnalyzedStatement::While(_) => todo!(),
            AnalyzedStatement::For(_) => todo!(),
            AnalyzedStatement::Break => todo!(),
            AnalyzedStatement::Continue => todo!(),
            AnalyzedStatement::Expr(node) => {
                self.expression(node);
            }
        }
    }

    /// Allocates the variable on the stack.
    /// Also increments the `additional_stack_space` of the current function
    fn let_statement(&mut self, node: AnalyzedLetStmt) {
        let value_reg = self.expression(node.expr);
        // store the value of the expr on the stack
        match value_reg {
            Some(Register::Int(reg)) => self.insert(Instruction::Sd(
                reg,
                Pointer::Stack(IntRegister::Fp, self.curr_fn.fp as i64),
            )),
            Some(Register::Float(reg)) => self.insert(Instruction::Fsd(FldType::Stack(
                reg,
                IntRegister::T0,
                self.curr_fn.fp as i64,
            ))),
            None => {
                // insert a dummy variable into the HashMap
                self.scopes
                    .last_mut()
                    .expect("there must be a scope")
                    .insert(node.name.to_string(), None);
            }
        }

        // insert variable into current scope
        self.scopes
            .last_mut()
            .expect("there must be a scope")
            .insert(node.name.to_string(), Some(self.curr_fn.fp as i64));

        // increment frame pointer
        self.curr_fn.fp += 8;
    }

    fn expression(&mut self, node: AnalyzedExpression) -> Option<Register> {
        match node {
            AnalyzedExpression::Block(node) => self.block(*node),
            AnalyzedExpression::If(_) => todo!(),
            AnalyzedExpression::Int(value) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Li(dest_reg, value));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Float(value) => {
                let dest_reg = self.alloc_freg();
                // add a float constant with the value
                let label = format!("float_constant_{}", self.rodata_section.len());
                self.rodata_section.push(DataObj {
                    label: label.clone(),
                    data: DataObjType::Float(value),
                });
                // load value from float constant into `dest_reg`
                self.insert(Instruction::Fld(FldType::Label(
                    dest_reg,
                    label,
                    self.alloc_ireg(),
                )));
                Some(Register::Float(dest_reg))
            }
            AnalyzedExpression::Bool(value) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Li(dest_reg, value as i64));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Char(value) => {
                let dest_reg = self.alloc_ireg();
                self.insert(Instruction::Li(dest_reg, value as i64));
                Some(Register::Int(dest_reg))
            }
            AnalyzedExpression::Ident(node) => {
                let ptr = self
                    .scopes
                    .last()
                    .expect("there is a scope")
                    .get(node.ident)
                    .expect("variable exits");

                match ptr {
                    Some(offset) => match node.result_type {
                        Type::Bool | Type::Char => {
                            let dest_reg = self.alloc_ireg();
                            self.insert(Instruction::Lb(
                                dest_reg,
                                Pointer::Stack(IntRegister::Sp, *offset),
                            ));
                            Some(Register::Int(dest_reg))
                        }
                        Type::Int => {
                            let dest_reg = self.alloc_ireg();
                            self.insert(Instruction::Ld(
                                dest_reg,
                                Pointer::Stack(IntRegister::Sp, *offset),
                            ));
                            Some(Register::Int(dest_reg))
                        }
                        Type::Float => {
                            let dest_reg = self.alloc_freg();
                            self.insert(Instruction::Fld(FldType::Stack(
                                dest_reg,
                                self.alloc_ireg(),
                                *offset,
                            )));
                            Some(Register::Float(dest_reg))
                        }
                        Type::Unit | Type::Never => None,
                        Type::Unknown => unreachable!("analyzer would have failed"),
                    },
                    None => None,
                }
            }
            AnalyzedExpression::Prefix(_) => todo!(),
            AnalyzedExpression::Infix(_) => todo!(),
            AnalyzedExpression::Assign(_) => todo!(),
            AnalyzedExpression::Call(node) => self.call_expr(*node),
            AnalyzedExpression::Cast(_) => todo!(),
            AnalyzedExpression::Grouped(node) => self.expression(*node),
        }
    }

    fn call_expr(&mut self, node: AnalyzedCallExpr) -> Option<Register> {
        // initial argument register
        let mut curr_int_reg = IntRegister::A0;
        let mut curr_flt_reg = FloatRegister::Fa0;

        for arg in node
            .args
            .into_iter()
            // skip any unit or never values
            .filter(|a| !matches!(a.result_type(), Type::Unit | Type::Never))
        {
            match arg.result_type() {
                Type::Int | Type::Char | Type::Bool => {
                    // TODO: is curr used?
                    // TODO: push on stack, save in vec

                    // expr result will be in correct reg
                    self.expression(arg).expect("filtered out above");
                    // mark the curr_reg as used
                    self.used_registers.push(curr_int_reg);
                    match curr_int_reg.next_param() {
                        Some(next) => curr_int_reg = next,
                        None => todo!("stack args are not yet implemented"),
                    }
                }
                Type::Float => {
                    // TODO: is curr used?
                    // TODO: push on stack, save in vec

                    // expr result will be in correct reg
                    self.expression(arg).expect("filtered out above");
                    // mark the curr_reg as used
                    self.used_float_registers.push(curr_flt_reg);
                    match curr_flt_reg.next_param() {
                        Some(next) => curr_flt_reg = next,
                        None => todo!("stack args are not yet implemented"),
                    }
                }
                Type::Unit | Type::Never => {} // ignore these types
                Type::Unknown => unreachable!("cannot use values of type `{{unknown}}` as params"),
            }
        }
        // TODO: restore all previously pushed from vec?

        // perform the actual call
        self.insert(Instruction::Call(node.func.into()));

        // return the call return value
        match node.result_type {
            Type::Unit | Type::Never => None,
            Type::Unknown => unreachable!("the analyze would have failed"),
            _ => Some(Register::Int(IntRegister::A0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, process::Command};

    use super::*;

    #[test]
    fn test_compiler() {
        let path = "./test.rush";
        let code = fs::read_to_string(path).unwrap();
        let (ast, _) = rush_analyzer::analyze(&code, path).unwrap();
        let mut compiler = Compiler::new();
        let out = compiler.compile(ast);
        fs::write("test.s", out).unwrap();

        Command::new("riscv64-linux-gnu-gcc")
            .args([
                "-nostdlib",
                "-static",
                "test.s",
                "-L",
                "corelib",
                "-lcore",
                "-o",
                "test",
            ])
            .arg("-static")
            .output()
            .unwrap();
    }
}

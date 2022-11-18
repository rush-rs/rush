use rush_analyzer::ast::*;

use crate::{
    instruction::{Instruction, Section},
    register::IntRegister,
    value::{FloatValue, IntValueOrImm, Offset, Size, Value},
};

#[derive(Debug, Default)]
pub struct Compiler {
    function_body: Vec<Instruction>,

    global_symbols: Vec<Instruction>,
    text_section: Vec<Instruction>,

    //////// .data section ////////
    quad_globals: Vec<Instruction>,
    // long_globals: Vec<Instruction>,
    // short_globals: Vec<Instruction>,
    byte_globals: Vec<Instruction>,

    //////// .rodata section ////////
    octa_constants: Vec<Instruction>,
    quad_constants: Vec<Instruction>,
    // long_constants: Vec<Instruction>,
    // short_constants: Vec<Instruction>,
    // byte_constants: Vec<Instruction>,
}

impl<'src> Compiler {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn compile(mut self, tree: AnalyzedProgram<'src>) -> String {
        self.program(tree);

        let mut buf = vec![Instruction::IntelSyntax];
        buf.append(&mut self.global_symbols);

        buf.push(Instruction::Section(Section::Text));
        buf.append(&mut self.text_section);

        if !self.quad_globals.is_empty() || !self.byte_globals.is_empty() {
            buf.push(Instruction::Section(Section::Data));
            buf.append(&mut self.quad_globals);
            // buf.append(&mut self.long_globals);
            // buf.append(&mut self.short_globals);
            buf.append(&mut self.byte_globals);
        }

        if !self.octa_constants.is_empty() || !self.quad_constants.is_empty() {
            buf.push(Instruction::Section(Section::ReadOnlyData));
            buf.append(&mut self.octa_constants);
            buf.append(&mut self.quad_constants);
            // buf.append(&mut self.long_constants);
            // buf.append(&mut self.short_constants);
            // buf.append(&mut self.byte_constants);
        }

        buf.into_iter().map(|instr| instr.to_string()).collect()
    }

    /////////////////////////////////////////

    fn program(&mut self, node: AnalyzedProgram<'src>) {
        self.main_fn(node.main_fn);
    }

    fn main_fn(&mut self, body: AnalyzedBlock<'src>) {
        self.text_section.push(Instruction::Symbol("_start".into()));
        self.global_symbols
            .push(Instruction::Global("_start".into()));
        self.function_body(body);
    }

    fn function_body(&mut self, node: AnalyzedBlock<'src>) {
        for stmt in node.stmts {
            self.statement(stmt);
        }

        if let Some(expr) = node.expr {
            self.expression(expr);
        }

        self.text_section.append(&mut self.function_body);
    }

    /////////////////////////////////////////

    fn statement(&mut self, node: AnalyzedStatement<'src>) {
        match node {
            AnalyzedStatement::Let(_) => todo!(),
            AnalyzedStatement::Return(_) => todo!(),
            AnalyzedStatement::Loop(_) => todo!(),
            AnalyzedStatement::While(_) => todo!(),
            AnalyzedStatement::For(_) => todo!(),
            AnalyzedStatement::Break => todo!(),
            AnalyzedStatement::Continue => todo!(),
            AnalyzedStatement::Expr(expr) => {
                self.expression(expr);
            }
        }
    }

    /////////////////////////////////////////

    fn expression(&mut self, node: AnalyzedExpression<'src>) -> Value {
        match node {
            AnalyzedExpression::Block(_node) => todo!(),
            AnalyzedExpression::If(_node) => todo!(),
            AnalyzedExpression::Int(num) => Value::IntOrImm(IntValueOrImm::Immediate(num)),
            AnalyzedExpression::Float(num) => {
                let symbol_name = format!(".QuadConstant_{}", self.quad_constants.len() / 2);
                self.quad_constants
                    .push(Instruction::Symbol(symbol_name.clone()));
                self.quad_constants.push(Instruction::Quad(num.to_bits()));
                Value::Float(FloatValue::Ptr(
                    Size::Qword,
                    IntRegister::Rip,
                    Offset::Symbol(symbol_name),
                ))
            }
            AnalyzedExpression::Bool(bool) => {
                Value::IntOrImm(IntValueOrImm::Immediate(bool as i64))
            }
            AnalyzedExpression::Char(num) => Value::IntOrImm(IntValueOrImm::Immediate(num as i64)),
            AnalyzedExpression::Ident(_node) => todo!(),
            AnalyzedExpression::Prefix(_node) => todo!(),
            AnalyzedExpression::Infix(_node) => todo!(),
            AnalyzedExpression::Assign(_node) => todo!(),
            AnalyzedExpression::Call(_node) => todo!(),
            AnalyzedExpression::Cast(_node) => todo!(),
            AnalyzedExpression::Grouped(node) => self.expression(*node),
        }
    }
}

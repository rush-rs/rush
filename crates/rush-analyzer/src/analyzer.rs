use std::collections::HashMap;

use rush_parser::{
    ast::{FunctionDefinition, Program, Type},
    Span,
};

use crate::Diagnostic;

pub struct Analyzer<'src> {
    program: Program<'src>,
    pub functions: Vec<FunctionDefinition<'src>>,
    scopes: Vec<Scope<'src>>,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Default)]
pub struct Scope<'src> {
    pub vars: HashMap<&'src str, Variable>,
}

#[derive(Debug)]
pub struct Variable {
    pub type_: Type,
    pub span: Span,
}

impl<'src> Analyzer<'src> {
    pub fn new(program: Program<'src>) -> Self {
        Self {
            program,
            functions: vec![],
            scopes: vec![Scope::default()],
            diagnostics: vec![],
        }
    }
}

#[macro_use]
mod macros;

mod analyzer;
pub mod ast;
mod diagnostic;

use std::iter;

pub use analyzer::*;
use ast::AnalyzedProgram;
pub use diagnostic::*;

pub use rush_parser::ast::Type;
use rush_parser::{Lexer, Parser};

/// Analyzes rush source code and returns an analyzed (annotated) AST.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// However, the `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error
pub fn analyze(text: &str) -> Result<(AnalyzedProgram, Vec<Diagnostic>), Vec<Diagnostic>> {
    let lexer = Lexer::new(text);

    let parser = Parser::new(lexer);
    let (ast, errs) = parser.parse();

    match (ast, errs.len()) {
        (Err(critical), _) => Err(errs
            .into_iter()
            .map(Diagnostic::from)
            .chain(iter::once(critical.into()))
            .collect()),
        (Ok(ast), _) => {
            let analyzer = Analyzer::new();

            // saves potential issues of the parser as diagnostics
            let mut parser_diagnostics: Vec<Diagnostic> =
                errs.into_iter().map(Diagnostic::from).collect();

            let (analyzed_ast, mut analyzer_diagnostics) = match analyzer.analyze(ast) {
                Ok(res) => res,
                Err(mut analyzer_diagnostics) => {
                    parser_diagnostics.append(&mut analyzer_diagnostics);
                    return Err(parser_diagnostics);
                }
            };

            // append the analyzer diagnostics to the parser errors
            parser_diagnostics.append(&mut analyzer_diagnostics);

            // return the `Err(_)` variant if the diagnostics contain at least 1 error
            match parser_diagnostics
                .iter()
                .any(|diagnostic| matches!(diagnostic.level, DiagnosticLevel::Error(_)))
            {
                true => Err(parser_diagnostics),
                false => Ok((analyzed_ast, parser_diagnostics)),
            }
        }
    }
}

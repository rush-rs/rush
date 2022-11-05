mod analyzer;
mod ast;
mod diagnostic;

use std::iter;

pub use analyzer::*;
use ast::AnalyzedProgram;
pub use diagnostic::*;
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
        (Ok(ast), 0) => {
            let analyzer = Analyzer::new();
            let (analyzed_ast, diagnostics) = analyzer.analyze(ast);

            match diagnostics
                .iter()
                .any(|diagnostic| matches!(diagnostic.level, DiagnosticLevel::Error(_)))
            {
                true => Err(diagnostics),
                false => Ok((analyzed_ast, diagnostics)),
            }
        }
        (Ok(_), _) => Err(errs.into_iter().map(Diagnostic::from).collect()),
    }
}

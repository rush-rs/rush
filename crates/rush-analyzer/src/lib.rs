mod analyzer;
mod ast;
mod diagnostic;

pub use analyzer::*;
pub use diagnostic::*;
use rush_parser::{Lexer, Parser};

/// Analyzes rush source code and returns an analyzed (annotated) AST.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// However, the `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error
pub fn analyze<'src>(
    text: &'src str,
) -> Result<(AnalyzedProgram<'src>, Vec<Diagnostic>), Vec<Diagnostic>> {
    let lexer = Lexer::new(text);

    let parser = Parser::new(lexer);
    let (ast, errs) = parser.parse();

    match (ast, errs.len()) {
        (Err(critical), _) => Err(errs
            .iter()
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
        (Ok(_), _) => Err(errs.iter().map(Diagnostic::from).collect()),
    }
}

use std::process;

use rush_analyzer::{ast::AnalyzedProgram, Diagnostic};

/// Analyzes the specified source code and prints out any diagnostics.
/// All diagnostics are printed to stderr and the process is exited when there is an error.
pub fn analyze<'src>(text: &'src str, path: &'src str) -> AnalyzedProgram<'src> {
    let ast = match rush_analyzer::analyze(text, path) {
        Ok((ast, diagnostics)) => {
            print_diagnostics(&diagnostics);
            ast
        }
        Err(diagnostics) => {
            print_diagnostics(&diagnostics);
            process::exit(1);
        }
    };
    ast
}

#[inline]
fn print_diagnostics(diagnostics: &[Diagnostic]) {
    for diag in diagnostics {
        eprintln!("{diag:#}");
    }
}

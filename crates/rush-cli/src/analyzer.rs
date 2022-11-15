use anyhow::{anyhow, Result};
use rush_analyzer::{ast::AnalyzedProgram, Diagnostic, DiagnosticLevel};

/// Analyzes the specified source code and prints out any diagnostics.
/// If an error occurs, the first error's kind is returned as the `Err(_)` variant.
pub fn analyze<'src>(text: &'src str, path: &'src str) -> Result<AnalyzedProgram<'src>> {
    let ast = match rush_analyzer::analyze(text, path) {
        Ok((ast, diagnostics)) => {
            print_diagnostics(&diagnostics);
            ast
        }
        Err(diagnostics) => {
            print_diagnostics(&diagnostics);
            let err = diagnostics
                .iter()
                .find_map(|d| {
                    if let DiagnosticLevel::Error(err) = &d.level {
                        Some(err.to_string())
                    } else {
                        None
                    }
                })
                .expect("there is always an error");
            return Err(anyhow!(err));
        }
    };
    Ok(ast)
}

fn print_diagnostics(diagnostics: &[Diagnostic]) {
    let display = diagnostics
        .iter()
        .map(|d| format!("{d:#}\n"))
        .collect::<Vec<String>>()
        .join("");
    print!("{display}");
}

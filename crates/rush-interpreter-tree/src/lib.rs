mod interpreter;
mod ops;
mod value;

pub use interpreter::*;
use rush_analyzer::Diagnostic;
pub use value::*;

/// Interprets rush source code by walking the analyzed tree.
/// The `Ok(_)` variant returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn run<'src>(
    text: &'src str,
    path: &'src str,
) -> Result<Vec<Diagnostic<'src>>, Vec<Diagnostic<'src>>> {
    let (tree, diagnostics) = rush_analyzer::analyze(text, path)?;
    Interpreter::new().run(tree);
    Ok(diagnostics)
}

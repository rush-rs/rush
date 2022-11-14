pub use compiler::Compiler;
use rush_analyzer::Diagnostic;

mod compiler;
mod corelib;
mod instructions;
mod types;
mod utils;

/// Compiles rush source code to a binary WebAssembly module.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn compile<'src>(
    text: &'src str,
    path: &'src str,
) -> Result<(Vec<u8>, Vec<Diagnostic<'src>>), Vec<Diagnostic<'src>>> {
    let (tree, diagnostics) = rush_analyzer::analyze(text, path)?;
    let bin = Compiler::new().compile(tree);
    Ok((bin, diagnostics))
}

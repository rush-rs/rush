pub use compiler::Compiler;
use rush_analyzer::Diagnostic;

mod compiler;
mod corelib;
mod instructions;
mod types;
mod utils;

pub type CompileResult<'src> = Result<(Vec<u8>, Vec<Diagnostic<'src>>), Vec<Diagnostic<'src>>>;

/// Compiles rush source code to a binary WebAssembly module.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn compile<'src>(text: &'src str, path: &'src str) -> Result<CompileResult<'src>, String> {
    let (tree, diagnostics) = match rush_analyzer::analyze(text, path) {
        Ok(tree) => tree,
        Err(diagnostics) => return Ok(Err(diagnostics)),
    };
    let bin = Compiler::new().compile(tree)?;
    Ok(Ok((bin, diagnostics)))
}

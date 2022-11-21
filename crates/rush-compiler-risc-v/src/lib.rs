use compiler::Compiler;
use rush_analyzer::Diagnostic;

mod call;
mod compiler;
mod corelib;
mod instruction;
mod register;
mod utils;

/// Compiles rush source code to a RISC-V assembly.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn compile<'src>(
    text: &'src str,
    path: &'src str,
) -> Result<(String, Vec<Diagnostic<'src>>), Vec<Diagnostic<'src>>> {
    let (tree, diagnostics) = rush_analyzer::analyze(text, path)?;
    let asm = Compiler::new().compile(tree);
    Ok((asm, diagnostics))
}

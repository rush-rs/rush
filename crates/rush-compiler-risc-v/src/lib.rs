pub use compiler::Compiler;
pub use instruction::CommentConfig;
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
pub fn compile<'tree>(
    text: &'tree str,
    path: &'tree str,
    comment_config: &CommentConfig,
) -> Result<(String, Vec<Diagnostic<'tree>>), Vec<Diagnostic<'tree>>> {
    let (tree, diagnostics) = rush_analyzer::analyze(text, path)?;
    let asm = Compiler::new().compile(tree, comment_config);
    Ok((asm, diagnostics))
}

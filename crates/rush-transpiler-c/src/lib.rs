use rush_analyzer::Diagnostic;
pub use transpiler::Transpiler;

mod c_ast;
mod transpiler;

/// Transpiles rush source code to C89 / C90.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn transpile<'tree>(
    text: &'tree str,
    path: &'tree str,
    emit_comments: bool,
) -> Result<(String, Vec<Diagnostic<'tree>>), Vec<Diagnostic<'tree>>> {
    let (tree, diagnostics) = rush_analyzer::analyze(text, path)?;
    let c_ast = Transpiler::new(emit_comments).transpile(tree);
    Ok((c_ast.to_string(), diagnostics))
}

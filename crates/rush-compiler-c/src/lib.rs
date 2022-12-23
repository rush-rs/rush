use rush_analyzer::Diagnostic;
use transpiler::Transpiler;

mod transpiler;

/// Transpiles rush source code to C89 / C90.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn transpile<'tree>(
    text: &'tree str,
    path: &'tree str,
) -> Result<(String, Vec<Diagnostic<'tree>>), Vec<Diagnostic<'tree>>> {
    let (tree, diagnostics) = rush_analyzer::analyze(text, path)?;
    let c_code = Transpiler::new().transpile(tree);
    Ok((c_code, diagnostics))
}

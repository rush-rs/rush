pub use compiler::Compiler;
use rush_analyzer::Diagnostic;

mod compiler;

/// Compiles rush source code to a binary WebAssembly module.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn compile(text: &str) -> Result<(Vec<u8>, Vec<Diagnostic>), Vec<Diagnostic>> {
    let (tree, diagnostics) = rush_analyzer::analyze(text)?;
    let bin = Compiler::new().compile(tree);
    Ok((bin, diagnostics))
}

#[cfg(test)]
mod tests {
    use std::fs;

    #[test]
    fn compile() {
        let bytes = crate::compile(
            r#"
fn main() {
    1 + 2;
}
"#,
        )
        .unwrap()
        .0;
        fs::write("output.wasm", bytes).unwrap();
    }
}

pub use compiler::Compiler;
use rush_analyzer::Diagnostic;

mod compiler;
mod instructions;
mod types;
mod utils;

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
    'a';
    return;
}

fn _add(left: int) -> int {
    let mut right = 4;
    right += 6;
    return left / right;
}
"#,
        )
        .unwrap()
        .0;
        fs::write("output.wasm", bytes).unwrap();
    }
}

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
    let a: int = 0x80;

    if a as char == '\x00' { // this should be false
        // exit(1);
    } else if a as char == '\x7F' { // this should be true
        // exit(2);
    } else {
        // exit(3);
    }
}

/*
// fn _add(left: int) -> int {
//     let mut right = 4.0;
//     right += 6.5;
//     return left / right as int;
// }
*/

fn _other() -> char {
    if false { return 'c'; }
    if 2 > 1 { 'a' } else { 'b' }
}
"#,
        )
        .unwrap()
        .0;
        fs::write("output.wasm", bytes).unwrap();
    }
}

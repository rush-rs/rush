mod compiler;
mod instruction;
mod value;
mod vm;

use compiler::Compiler;
pub use instruction::Instruction;
use rush_analyzer::ast::AnalyzedProgram;
pub use rush_analyzer::Diagnostic;
pub use value::Value;
use vm::{RuntimeError, Vm};

/// Holds the result of a successful compilation.
/// Includes VM instructions and non-error diagnostics.
pub struct CompilationResult<'src> {
    pub program: Vec<Vec<Instruction>>,
    pub diagnostics: Vec<Diagnostic<'src>>,
}

/// Compiles rush source code to VM instructions.
/// The `Ok(_)` variant is a [`CompilationResult`] which also includes any non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn compile<'src>(
    text: &'src str,
    path: &'src str,
) -> Result<CompilationResult<'src>, Vec<Diagnostic<'src>>> {
    let (tree, diagnostics) = rush_analyzer::analyze(text, path)?;
    let instr = Compiler::new().compile(&tree);
    Ok(CompilationResult {
        program: instr,
        diagnostics,
    })
}

/// Executes the given program using the VM.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn run(ast: AnalyzedProgram) -> Result<i64, RuntimeError> {
    let program = Compiler::new().compile(&ast);
    let mut vm = Vm::new();
    let exit_code = vm.run(program)?;
    Ok(exit_code)
}

/// Executes the given program using the VM on debug mode.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn debug_run(ast: AnalyzedProgram, clock_hz: u64) -> Result<i64, RuntimeError> {
    let program = Compiler::new().compile(&ast);
    let mut vm = Vm::new();
    let exit_code = vm.debug_run(program, clock_hz)?;
    Ok(exit_code)
}

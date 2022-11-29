mod compiler;
mod instruction;
mod value;
mod vm;

use compiler::Compiler;
pub use instruction::Instruction;
use instruction::Program;
use rush_analyzer::ast::AnalyzedProgram;
pub use rush_analyzer::Diagnostic;
pub use value::Value;
use vm::{RuntimeError, Vm};

/// Compiles rush source code to VM instructions.
/// The `Ok(_)` variant is a [`CompilationResult`] which also includes any non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn compile(ast: &AnalyzedProgram) -> Program {
    Compiler::new().compile(ast)
}

/// Executes the given program using the VM.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn run(ast: &AnalyzedProgram) -> Result<i64, RuntimeError> {
    let program = Compiler::new().compile(ast);
    let mut vm = Vm::new();
    let exit_code = vm.run(program.0)?;
    Ok(exit_code)
}

/// Executes the given program using the VM on debug mode.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn debug_run(ast: &AnalyzedProgram, clock_hz: u64) -> Result<i64, RuntimeError> {
    let program = Compiler::new().compile(ast);
    let mut vm = Vm::new();
    let exit_code = vm.debug_run(program.0, clock_hz)?;
    Ok(exit_code)
}

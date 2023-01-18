#![warn(rust_2018_idioms)]

mod compiler;
mod instruction;
mod value;
mod vm;

pub use compiler::Compiler;
pub use instruction::Instruction;
use instruction::Program;
use rush_analyzer::ast::AnalyzedProgram;
pub use rush_analyzer::Diagnostic;
pub use value::Value;
pub use vm::{RuntimeError, RuntimeErrorKind, Vm};

/// Compiles rush source code to VM instructions.
/// The `Ok(_)` variant is a [`CompilationResult`] which also includes any non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn compile(ast: AnalyzedProgram<'_>) -> Program {
    Compiler::new().compile(ast)
}

/// Executes the given program using the VM.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn run<const MEM_SIZE: usize>(ast: AnalyzedProgram<'_>) -> Result<i64, RuntimeError> {
    let program = Compiler::new().compile(ast);
    let mut vm: Vm<MEM_SIZE> = Vm::new();
    let exit_code = vm.run(program)?;
    Ok(exit_code)
}

/// Executes the given program using the VM on debug mode.
/// The `Ok(_)` variant also returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn debug_run<const MEM_SIZE: usize>(
    ast: AnalyzedProgram<'_>,
    clock_hz: u64,
) -> Result<i64, RuntimeError> {
    let program = Compiler::new().compile(ast);
    let mut vm: Vm<MEM_SIZE> = Vm::new();
    let exit_code = vm.debug_run(program, clock_hz)?;
    Ok(exit_code)
}

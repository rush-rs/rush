mod compiler;
mod corelib;
mod error;

pub use compiler::*;
pub use error::*;
pub use inkwell::{
    context::Context,
    memory_buffer::MemoryBuffer,
    targets::{TargetMachine, TargetTriple},
    OptimizationLevel,
};
use rush_analyzer::ast::AnalyzedProgram;

pub fn compile(
    ast: AnalyzedProgram,
    target: TargetTriple,
    optimization: OptimizationLevel,
) -> Result<(MemoryBuffer, String)> {
    let context = Context::create();
    let mut compiler = Compiler::new(&context, target, optimization);
    compiler.compile(ast)
}

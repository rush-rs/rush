mod compiler;
mod corelib;
mod error;

pub use compiler::*;
pub use error::*;
pub use inkwell;
use inkwell::{
    context::Context, memory_buffer::MemoryBuffer, targets::TargetTriple, OptimizationLevel,
};
use rush_analyzer::ast::AnalyzedProgram;

/// Compiles a rush AST into LLVM IR and an object file.
/// The `main_fn` param specifies whether the entry is the main function or `_start`.
pub fn compile(
    ast: AnalyzedProgram,
    target: TargetTriple,
    optimization: OptimizationLevel,
    main_fn: bool,
) -> Result<(MemoryBuffer, String)> {
    let context = Context::create();
    let mut compiler = Compiler::new(&context, target, optimization);
    compiler.compile(&ast, main_fn)
}

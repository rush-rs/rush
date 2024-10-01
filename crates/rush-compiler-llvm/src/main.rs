use std::{env, fs};

use inkwell::{
    context::Context,
    targets::{TargetMachine, TargetTriple},
};
use rush_compiler_llvm::Compiler;

fn main() {
    let filename = env::args().nth(1).unwrap();
    let triple = env::args().nth(2).unwrap_or(
        TargetMachine::get_default_triple()
            .as_str()
            .to_string_lossy()
            .to_string(),
    );

    let file = fs::read_to_string(&filename).unwrap();
    let ast = match rush_analyzer::analyze(&file, &filename) {
        Ok(res) => {
            for diagnostic in &res.1 {
                println!("{diagnostic:#}");
            }
            res.0
        }
        Err(diagnostics) => {
            for diagnostic in diagnostics {
                println!("{diagnostic:#}");
            }
            panic!("Analyzer detected issues");
        }
    };

    let context = Context::create();
    let mut compiler = Compiler::new(
        &context,
        TargetTriple::create(triple.as_str()),
        inkwell::OptimizationLevel::None,
        true,
    )
    .unwrap();

    let (obj, asm, ir) = compiler.compile(&ast).unwrap();
    fs::write("./output.ll", ir).unwrap();
    fs::write("./output.s", asm.as_slice()).unwrap();
    fs::write("./output.o", obj.as_slice()).unwrap();
}

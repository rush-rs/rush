use std::{fs, path::PathBuf};

use rush_analyzer::ast::AnalyzedProgram;
use rush_compiler_risc_v::Compiler;

use crate::cli::BuildArgs;

pub fn compile(ast: AnalyzedProgram, args: BuildArgs) {
    let mut compiler = Compiler::new();
    let asm = compiler.compile(ast);

    // get output path
    let output = args.output_file.unwrap_or_else(|| {
        let mut path = PathBuf::from(
            args.path
                .file_stem()
                .expect("file reading would have failed before"),
        );
        path.set_extension("s");
        path
    });

    fs::write(&output, asm)
        .unwrap_or_else(|err| eprintln!("cannot write to `{}`: {err}", output.to_string_lossy()));
}

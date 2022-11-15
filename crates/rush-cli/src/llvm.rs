use std::{
    fs,
    path::{Path, PathBuf},
    process::{self, Command, Stdio},
};

use rush_analyzer::ast::AnalyzedProgram;
use rush_compiler_llvm::{OptimizationLevel, TargetMachine};

pub fn compile(ast: AnalyzedProgram, opt: OptimizationLevel, show_ir: bool, output: &PathBuf) {
    let (obj, ir) = rush_compiler_llvm::compile(ast, TargetMachine::get_default_triple(), opt)
        .unwrap_or_else(|err| {
            eprintln!("compilation failed: llvm error: {err}");
            process::exit(1);
        });

    if show_ir {
        println!("{ir}");
    }
    // write to file
    let mut out = Path::new("/tmp/").join("rush.o");
    out.set_extension("o");

    fs::write(&out, obj.as_slice())
        .unwrap_or_else(|err| eprintln!("cannot write to `{}`: {err}", out.to_string_lossy()));

    // invoke gcc to link the file
    let command = Command::new("gcc")
        .arg(&out)
        .arg("-o")
        .arg(output)
        .stderr(Stdio::inherit())
        .output()
        .unwrap_or_else(|err| {
            eprintln!("could not invoke gcc: {err}");
            process::exit(1);
        });

    if !command.status.success() {
        eprintln!(
            "gcc failed with exit-code {}",
            command.status.code().unwrap()
        );
        process::exit(1);
    }
}

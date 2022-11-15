use std::{
    fs,
    path::{Path, PathBuf},
    process::{self, Command, Stdio},
    str::FromStr,
};

use analyzer::analyze;
use clap::Parser;
use cli::{Args, Backend, Command as ClapCommand};
use rush_analyzer::{ast::AnalyzedProgram, Diagnostic};
use rush_compiler_llvm::{OptimizationLevel, TargetMachine};
//use rush_compiler_wasm::Compiler as WasmCompiler;

mod analyzer;
mod cli;

fn main() {
    let args = Args::parse();

    match args.command {
        ClapCommand::Build {
            backend,
            output_file,
            llvm_opt,
            llvm_target,
            llvm_show_ir,
            file,
        } => {
            let path_str = file.to_string_lossy();
            let code = fs::read_to_string(&file).unwrap_or_else(|err| {
                eprintln!("cannot read `{}`: {err}", path_str);
                process::exit(1);
            });
            let ast = analyze(&code, &path_str).unwrap_or_else(|err| {
                eprintln!("compilation failed: {err}");
                process::exit(1);
            });

            match backend {
                Backend::Llvm => compile_llvm(
                    ast,
                    OptimizationLevel::from(llvm_opt),
                    llvm_show_ir,
                    &output_file.unwrap_or_else(|| {
                        { PathBuf::from_str(&file.file_stem().unwrap().to_string_lossy()) }.unwrap()
                    }),
                ),
                Backend::Wasm => {
                    todo!()
                }
            }
        }
        ClapCommand::Run { backend, file } => todo!("Run"),
        ClapCommand::Check { file } => {
            let path_str = file.to_string_lossy();
            let code = fs::read_to_string(&file).unwrap_or_else(|err| {
                eprintln!("cannot read `{}`: {err}", path_str);
                process::exit(1);
            });
            let ast = analyze(&code, &path_str).unwrap_or_else(|err| {
                eprintln!("compilation failed: {err}");
                process::exit(1);
            });
        }
    }
}

fn print_diagnostics(diagnostics: &[Diagnostic]) {
    println!(
        "{}",
        diagnostics
            .iter()
            .map(|d| format!("{d:#}\n"))
            .collect::<Vec<String>>()
            .join("")
    );
}

fn compile_llvm(ast: AnalyzedProgram, opt: OptimizationLevel, show_ir: bool, output: &PathBuf) {
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

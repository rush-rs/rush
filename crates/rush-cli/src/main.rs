use std::{fs, path::PathBuf, process, str::FromStr};

use analyzer::analyze;
use clap::Parser;
use cli::{Args, Backend, Command as ClapCommand};
use rush_compiler_llvm::OptimizationLevel;
//use rush_compiler_wasm::Compiler as WasmCompiler;

mod analyzer;
mod cli;
mod llvm;

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
                Backend::Llvm => llvm::compile(
                    ast,
                    OptimizationLevel::from(llvm_opt),
                    llvm_show_ir,
                    &output_file.unwrap_or_else(|| {
                        PathBuf::from_str(&file.file_stem().unwrap().to_string_lossy()).unwrap()
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

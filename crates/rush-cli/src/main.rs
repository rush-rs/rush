use std::{fs, process};

use clap::Parser;
use cli::{Backend, Cli};

mod analyzer;
mod cli;
mod llvm;

fn main() {
    match Cli::parse() {
        Cli::Build(args) => {
            let path_str = args.path.clone();
            let path_str = path_str.to_string_lossy();
            let code = fs::read_to_string(&args.path).unwrap_or_else(|err| {
                eprintln!("cannot read `{}`: {err}", path_str);
                process::exit(1);
            });
            let ast = analyzer::analyze(&code, &path_str);

            match args.backend {
                Backend::Llvm => llvm::compile(ast, args),
                Backend::Wasm => todo!(),
                Backend::RiscV => todo!(),
                Backend::X86_64 => todo!(),
            }
        }
        Cli::Run(args) => {
            let path_str = args.path.clone();
            let path_str = path_str.to_string_lossy();
            let code = fs::read_to_string(&args.path).unwrap_or_else(|err| {
                eprintln!("cannot read `{}`: {err}", path_str);
                process::exit(1);
            });
            let ast = analyzer::analyze(&code, &path_str);

            match args.backend {
                Backend::Llvm => llvm::run(ast, args),
                Backend::Wasm => todo!(),
                Backend::RiscV => todo!(),
                Backend::X86_64 => todo!(),
            }
        }
        Cli::Check { file } => {
            let path_str = file.to_string_lossy();
            let code = fs::read_to_string(&file).unwrap_or_else(|err| {
                eprintln!("cannot read `{}`: {err}", path_str);
                process::exit(1);
            });
            analyzer::analyze(&code, &path_str);
        }
    }
}

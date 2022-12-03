use std::{fs, process};

use clap::Parser;
use cli::{Backend, Cli};
use rush_interpreter_tree::Interpreter;

mod analyzer;
mod cli;

mod llvm;
mod riscv;
mod wasm;
mod x86;

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
                Backend::Wasm => wasm::compile(ast, args),
                Backend::RiscV => riscv::compile(ast, args),
                Backend::X86_64 => x86::compile(ast, args),
                Backend::Vm => println!("{}", rush_interpreter_vm::compile(&ast)),
                Backend::TreeWalking => {
                    eprintln!("cannot compile using an interpreted backend");
                    process::exit(1)
                }
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
                Backend::RiscV | Backend::X86_64 | Backend::Wasm => {
                    eprintln!(
                        "backend `{}` can only be used for compilation",
                        args.backend
                    );
                    process::exit(1);
                }
                Backend::TreeWalking => match Interpreter::new().run(ast) {
                    Ok(code) => {
                        println!("interpreter exited with code {code}");
                        process::exit(code as i32)
                    }
                    Err(err) => {
                        println!("\x1b[1;31minterpreter crashed\x1b[1;0m: {err}");
                        process::exit(1)
                    }
                },
                Backend::Vm => match rush_interpreter_vm::run(&ast) {
                    Ok(code) => {
                        println!("vm exited with code {code}");
                        process::exit(code as i32)
                    }
                    Err(err) => {
                        println!("\x1b[1;31mvm crashed\x1b[1;0m: {err}");
                        process::exit(1)
                    }
                },
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

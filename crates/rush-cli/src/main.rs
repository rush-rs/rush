use std::{fs, process, time::Instant};

use anyhow::{bail, Context};
use clap::Parser;
use cli::{Cli, Command, CompilerBackend, LlvmOpt, RunnableBackend};
use rush_analyzer::{ast::AnalyzedProgram, Diagnostic};
use rush_interpreter_tree::Interpreter;

mod cli;

mod llvm;
mod riscv;
mod wasm;
mod x86;

fn main() -> anyhow::Result<()> {
    let root_args = Cli::parse();

    match root_args.command {
        Command::Build(args) => {
            if args.backend != CompilerBackend::Llvm {
                if args.llvm_show_ir {
                    bail!("cannot show llvm IR when not using LLVM backend")
                }
                if args.llvm_opt != LlvmOpt::None {
                    bail!("cannot set LLVM optimization level when not using LLVM backend")
                }
                if args.llvm_target.is_some() {
                    bail!("cannot set LLVM target when not using LLVM backend")
                }
            }

            let compile_func = || -> anyhow::Result<()> {
                let total_start = Instant::now();
                let mut start = Instant::now();

                let text = fs::read_to_string(&args.path)?;

                let file_read_time = start.elapsed();
                start = Instant::now();

                let path = args.path.clone();
                let path = path.to_string_lossy();
                let tree = analyze(&text, &path)?;

                let analyze_time = start.elapsed();
                start = Instant::now();

                match args.backend {
                    CompilerBackend::Llvm => llvm::compile(tree, args)?,
                    CompilerBackend::Wasm => wasm::compile(tree, args)?,
                    CompilerBackend::RiscV => riscv::compile(tree, args)?,
                    CompilerBackend::X86_64 => {
                        x86::compile(tree, args)?;
                    }
                }

                if root_args.time {
                    println!("file read:        {file_read_time:?}");
                    println!("analyze:          {analyze_time:?}");
                    println!("compile:          {:?}", start.elapsed());
                    println!("\x1b[90mtotal:            {:?}\x1b[0m", total_start.elapsed());
                }

                Ok(())
            };
            compile_func().with_context(|| "compilation failed")?;
        }
        Command::Run(args) => {
            if args.backend != RunnableBackend::Llvm && args.llvm_opt != LlvmOpt::None {
                bail!("cannot set LLVM optimization level when not using LLVM backend")
            }

            let path = args.path.clone();
            let path = path.to_string_lossy();

            let run_func = || -> anyhow::Result<i64> {
                let total_start = Instant::now();
                let mut start = Instant::now();

                let text = fs::read_to_string(&args.path)?;

                let file_read_time = start.elapsed();
                start = Instant::now();

                let tree = analyze(&text, &path)?;

                let analyze_time = start.elapsed();
                start = Instant::now();

                let exit_code = match args.backend {
                    RunnableBackend::Tree => match Interpreter::new().run(tree) {
                        Ok(code) => code,
                        Err(err) => bail!(format!("interpreter crashed: {err}")),
                    },
                    RunnableBackend::Vm => match rush_interpreter_vm::run(tree) {
                        Ok(code) => code,
                        Err(err) => bail!(format!("vm crashed: {err}")),
                    },
                    RunnableBackend::Llvm => {
                        llvm::run(tree, args).with_context(|| "cannot run using `LLVM`")?
                    }
                    RunnableBackend::X86_64 => {
                        x86::run(tree, args).with_context(|| "cannot run using `x86_64`")?
                    }
                    RunnableBackend::RiscV => {
                        riscv::run(tree, args).with_context(|| "cannot run using `RISC-V`")?
                    }
                };

                if root_args.time {
                    println!("file read:            {file_read_time:?}");
                    println!("analyze:              {analyze_time:?}");
                    println!("run / compile:        {:?}", start.elapsed());
                    println!("\x1b[90mtotal:                {:?}\x1b[0m", total_start.elapsed());
                }

                Ok(exit_code)
            };

            let code = run_func().with_context(|| format!("running `{path}` failed",))?;
            process::exit(code as i32);
        }
        Command::Check { file: path } => {
            let check_func = || -> anyhow::Result<()> {
                let total_start = Instant::now();
                let mut start = Instant::now();

                let text = fs::read_to_string(&path)?;

                let file_read_time = start.elapsed();
                start = Instant::now();

                let path = path.to_string_lossy();
                analyze(&text, &path)?;

                if root_args.time {
                    println!("file read:        {file_read_time:?}");
                    println!("analyze:          {:?}", start.elapsed());
                    println!("\x1b[90mtotal:            {:?}\x1b[0m", total_start.elapsed());
                }

                Ok(())
            };

            check_func().with_context(|| {
                format!("checking `{file}` failed", file = path.to_string_lossy())
            })?;
        }
    }

    Ok(())
}

/// Analyzes the given rush source code, printing diagnostics alongside the way.
fn analyze<'src>(text: &'src str, path: &'src str) -> anyhow::Result<AnalyzedProgram<'src>> {
    match rush_analyzer::analyze(text, path) {
        Ok((program, diagnostics)) => {
            print_diagnostics(&diagnostics);
            Ok(program)
        }
        Err(diagnostics) => {
            print_diagnostics(&diagnostics);
            bail!("invalid program: analyzer detected issues")
        }
    }
}

#[inline]
/// Prints the given diagnostics to stderr.
fn print_diagnostics(diagnostics: &[Diagnostic]) {
    for d in diagnostics {
        eprintln!("{d:#}")
    }
}

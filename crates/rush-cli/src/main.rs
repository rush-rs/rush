use std::{fs, process, time::Instant};

use anyhow::{bail, Context};
use clap::Parser;
use cli::{Cli, Command, CompilerBackend, RunnableBackend};

#[cfg(feature = "llvm")]
use cli::LlvmOpt;
use rush_analyzer::{ast::AnalyzedProgram, Diagnostic};
use rush_interpreter_tree::Interpreter;

mod cli;

mod c;

#[cfg(feature = "llvm")]
mod llvm;

mod riscv;
mod wasm;
mod x86;

const VM_MEM_SIZE: usize = 10_024;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let root_args = Cli::parse();

    match root_args.command {
        Command::Build(args) => {
            #[cfg(feature = "llvm")]
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
                    #[cfg(feature = "llvm")]
                    CompilerBackend::Llvm => llvm::compile(tree, args)?,
                    CompilerBackend::Wasm => wasm::compile(tree, args)?,
                    CompilerBackend::RiscV => riscv::compile(tree, args, &tempfile::tempdir()?)?,
                    CompilerBackend::X86_64 => {
                        x86::compile(tree, args)?;
                    }
                    CompilerBackend::C => c::compile(tree, args)?,
                }

                if root_args.time {
                    eprintln!("file read:        {file_read_time:?}");
                    eprintln!("analyze:          {analyze_time:?}");
                    eprintln!("compile:          {:?}", start.elapsed());
                    eprintln!(
                        "\x1b[90mtotal:            {:?}\x1b[0m",
                        total_start.elapsed()
                    );
                }

                Ok(())
            };
            compile_func().with_context(|| "compilation failed")?;
        }
        Command::Run(args) => {
            #[cfg(feature = "llvm")]
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
                    RunnableBackend::Vm => match rush_interpreter_vm::run::<VM_MEM_SIZE>(tree) {
                        Ok(code) => code,
                        Err(err) => bail!(format!("vm crashed: {err}")),
                    },
                    #[cfg(feature = "llvm")]
                    RunnableBackend::Llvm => {
                        llvm::run(tree, args).with_context(|| "cannot run using `LLVM`")?
                    }
                    RunnableBackend::RiscV => {
                        riscv::run(tree, args).with_context(|| "cannot run using `RISC-V`")?
                    }
                    RunnableBackend::X86_64 => {
                        x86::run(tree, args).with_context(|| "cannot run using `x86_64`")?
                    }
                    RunnableBackend::C => c::run(tree, args)
                        .with_context(|| "cannot run using `ANSI C transpiler`")?,
                };

                if root_args.time {
                    eprintln!("file read:            {file_read_time:?}");
                    eprintln!("analyze:              {analyze_time:?}");
                    eprintln!("run / compile:        {:?}", start.elapsed());
                    eprintln!(
                        "\x1b[90mtotal:                {:?}\x1b[0m",
                        total_start.elapsed()
                    );
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
                    eprintln!("file read:        {file_read_time:?}");
                    eprintln!("analyze:          {:?}", start.elapsed());
                    eprintln!(
                        "\x1b[90mtotal:            {:?}\x1b[0m",
                        total_start.elapsed()
                    );
                }

                Ok(())
            };

            check_func().with_context(|| {
                format!("checking `{file}` failed", file = path.to_string_lossy())
            })?;
        }
        Command::Ls => rush_ls::start_service().await,
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

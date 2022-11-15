use std::path::PathBuf;

use clap::{ArgGroup, Parser, Subcommand, ValueEnum};
use rush_compiler_llvm::OptimizationLevel;

#[derive(Parser, Debug)]
#[clap(author, version, about)]
#[command(group(ArgGroup::new("llvm")))]
pub struct Args {
    /// Rush subcommands
    #[command(subcommand)]
    pub command: Command,
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum Backend {
    /// LLVM compiler: requires LLVM and gcc for compilation
    Llvm,
    /// WASM compiler: requires wasm runtime for later execution
    Wasm,
}

#[derive(Subcommand, PartialEq, Eq, Debug)]
pub enum Command {
    /// Build (b) a binary using the specified backend
    #[clap(alias = "b")]
    Build {
        /// Specifies the compiler backend
        #[clap(short, long, value_enum)]
        backend: Backend,

        /// Compiler output file
        #[clap(short, long, value_parser)]
        output_file: Option<PathBuf>,

        #[clap(short='O', long, value_parser, default_value=None)]
        llvm_opt: LlvmOpt,

        #[clap(long, value_parser)]
        llvm_target: Option<String>,

        #[clap(long, value_parser)]
        llvm_show_ir: bool,

        /// Rush Source file
        file: PathBuf,
    },
    /// Run (r) the source file's binary using the specified compiler
    #[clap(alias = "r")]
    Run {
        /// Specifies the compiler backend
        #[clap(short, long, value_enum)]
        backend: Backend,

        /// Rush Source file
        file: PathBuf,
    },
    /// Check (c) the source code without compilation
    #[clap(alias = "c")]
    Check {
        /// Rush Source file
        file: PathBuf,
    },
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum LlvmOpt {
    #[clap(name = "0")]
    /// No optimization
    None,
    #[clap(name = "1")]
    /// Less optimization
    Less,
    #[clap(name = "2")]
    /// Default optimization level
    Default,
    #[clap(name = "3")]
    /// Aggressive optimization
    Aggressive,
}

impl From<LlvmOpt> for OptimizationLevel {
    fn from(src: LlvmOpt) -> Self {
        match src {
            LlvmOpt::None => Self::None,
            LlvmOpt::Less => Self::Less,
            LlvmOpt::Default => Self::Default,
            LlvmOpt::Aggressive => Self::Aggressive,
        }
    }
}

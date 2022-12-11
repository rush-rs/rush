use std::{fmt::Display, path::PathBuf};

use clap::{Args, Parser, ValueEnum};
use rush_compiler_llvm::inkwell::OptimizationLevel;

#[derive(Parser, Debug)]
#[clap(author, version, about)]
pub enum Cli {
    /// Build a binary using the specified backend
    #[clap(alias = "b")]
    Build(BuildArgs),
    /// Run the source file's binary using the specified compiler
    #[clap(alias = "r")]
    Run(RunArgs),
    /// Check the source code without compilation
    #[clap(alias = "c")]
    Check {
        /// Rush Source file
        file: PathBuf,
    },
}

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Specifies the backend
    #[clap(short, long, value_enum)]
    pub backend: RunnableBackend,

    /// The optimization level when using the LLVM backend
    #[clap(short = 'O', long, value_parser, default_value = "0")]
    pub llvm_opt: LlvmOpt,

    /// Path to rush source file
    pub path: PathBuf,
}

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// Specifies the compiler backend
    #[clap(short, long, value_enum)]
    pub backend: CompilerBackend,

    /// Output file location
    #[clap(short, long, value_parser)]
    pub output_file: Option<PathBuf>,

    /// The optimization level when using the LLVM backend
    #[clap(short = 'O', long, value_parser, default_value = "0")]
    pub llvm_opt: LlvmOpt,

    /// The target triplet to use when using the LLVM backend, defaults to native
    #[clap(long, value_parser)]
    pub llvm_target: Option<String>,

    /// Print the generated LLVM IR to stdout
    #[clap(long, value_parser)]
    pub llvm_show_ir: bool,

    /// Path to rush source file
    pub path: PathBuf,
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum CompilerBackend {
    /// LLVM compiler: requires GCC
    Llvm,
    /// WASM compiler: requires wasm runtime for later execution
    Wasm,
    /// RISC-V compiler: requires RISC-V toolchain
    RiscV,
    /// X86_64 compiler: requires x86 toolchain
    X86_64,
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum RunnableBackend {
    /// Tree-walking interpreter: no dependencies
    Tree,
    /// VM interpreter: no dependencies
    Vm,
    /// LLVM compiler: requires GCC
    Llvm,
}

impl Display for CompilerBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CompilerBackend::Llvm => "llvm",
                CompilerBackend::Wasm => "wasm",
                CompilerBackend::RiscV => "risc-v",
                CompilerBackend::X86_64 => "x86_64",
            }
        )
    }
}

impl Display for RunnableBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RunnableBackend::Tree => "tree",
                RunnableBackend::Vm => "vm",
                RunnableBackend::Llvm => "llvm",
            }
        )
    }
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

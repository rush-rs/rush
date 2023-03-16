use std::{fmt::Display, path::PathBuf};

use anyhow::anyhow;
use clap::{Args, Parser, Subcommand, ValueEnum};
#[cfg(feature = "llvm")]
use rush_compiler_llvm::inkwell::OptimizationLevel;

#[derive(Parser, Debug)]
#[clap(author, version, about)]
pub struct Cli {
    /// rush Subcommands
    #[clap(subcommand)]
    pub command: Command,
    /// Enables time tracking for benchmarking
    #[clap(short, long, value_parser)]
    pub time: bool,
}

#[derive(Debug, Subcommand)]
pub enum Command {
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
    /// Launches the rush language server
    Ls,
}

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Specifies the backend
    #[clap(short, long, value_enum)]
    pub backend: RunnableBackend,

    /// The optimization level when using the LLVM backend
    #[cfg_attr(
        feature = "llvm",
        clap(short = 'O', long, value_parser, default_value = "0")
    )]
    #[cfg(feature = "llvm")]
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
    #[cfg_attr(
        feature = "llvm",
        clap(short = 'O', long, value_parser, default_value = "0")
    )]
    #[cfg(feature = "llvm")]
    pub llvm_opt: LlvmOpt,

    /// The target triplet to use when using the LLVM backend, defaults to native
    #[cfg_attr(feature = "llvm", clap(long, value_parser))]
    #[cfg(feature = "llvm")]
    pub llvm_target: Option<String>,

    /// Print the generated LLVM IR to stdout
    #[cfg_attr(feature = "llvm", clap(long, value_parser))]
    #[cfg(feature = "llvm")]
    pub llvm_show_ir: bool,

    /// Path to rush source file
    pub path: PathBuf,
}

impl TryFrom<RunArgs> for BuildArgs {
    type Error = anyhow::Error;

    fn try_from(args: RunArgs) -> Result<Self, Self::Error> {
        Ok(Self {
            backend: args.backend.try_into()?,
            output_file: None,
            #[cfg(feature = "llvm")]
            llvm_opt: args.llvm_opt,
            #[cfg(feature = "llvm")]
            llvm_target: None,
            #[cfg(feature = "llvm")]
            llvm_show_ir: false,
            path: args.path,
        })
    }
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum CompilerBackend {
    /// LLVM compiler: requires GCC
    #[cfg(feature = "llvm")]
    Llvm,
    /// WebAssembly compiler: requires wasm runtime for later execution
    Wasm,
    /// RISC-V compiler: requires RISC-V toolchain (alias = riscv)
    #[clap(alias = "riscv")]
    RiscV,
    /// X86_64 compiler: requires x86 toolchain (alias = x64)
    #[clap(alias = "x64")]
    X86_64,
    /// ANSI C transpiler: requires GCC
    C,
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum RunnableBackend {
    /// Tree-walking interpreter: no dependencies
    Tree,
    /// VM interpreter: no dependencies
    Vm,
    /// LLVM compiler: requires GCC
    #[cfg(feature = "llvm")]
    Llvm,
    /// RISC-V compiler: requires RISC-V toolchain (alias = riscv)
    #[clap(alias = "riscv")]
    RiscV,
    /// X86_64 compiler: requires x86 toolchain (alias = x64)
    #[clap(alias = "x64")]
    X86_64,
    /// ANSI C transpiler: requires GCC
    C,
}

impl TryFrom<RunnableBackend> for CompilerBackend {
    type Error = anyhow::Error;

    fn try_from(backend: RunnableBackend) -> Result<Self, Self::Error> {
        match backend {
            RunnableBackend::Tree | RunnableBackend::Vm => {
                Err(anyhow!("cannot use interpreter backends for compilation"))
            }
            #[cfg(feature = "llvm")]
            RunnableBackend::Llvm => Ok(CompilerBackend::Llvm),
            RunnableBackend::RiscV => Ok(CompilerBackend::RiscV),
            RunnableBackend::X86_64 => Ok(CompilerBackend::X86_64),
            RunnableBackend::C => Ok(CompilerBackend::C),
        }
    }
}

impl Display for CompilerBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                #[cfg(feature = "llvm")]
                CompilerBackend::Llvm => "llvm",
                CompilerBackend::Wasm => "wasm",
                CompilerBackend::RiscV => "risc-v",
                CompilerBackend::X86_64 => "x86_64",
                CompilerBackend::C => "c",
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
                #[cfg(feature = "llvm")]
                RunnableBackend::Llvm => "llvm",
                RunnableBackend::RiscV => "risc-v",
                RunnableBackend::X86_64 => "x86_64",
                RunnableBackend::C => "c",
            }
        )
    }
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
#[cfg(feature = "llvm")]
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

#[cfg(feature = "llvm")]
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

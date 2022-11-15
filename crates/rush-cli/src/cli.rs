use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[clap(author, version, about)]
pub struct Args {
    /// Specifies the compiler backend
    #[clap(short, long, value_enum)]
    pub backend: Backend,
    /// rush subcommands
    #[command(subcommand)]
    pub command: Command,
}

#[derive(ValueEnum, Clone, Debug)]
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
    Build,
    /// Run (r) the source file's binary using the specified compiler
    #[clap(alias = "r")]
    Run,
    /// Check (c) the source code without compilation
    #[clap(alias = "c")]
    Check,
}

use inkwell::{support::LLVMString, targets::TargetTriple};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Llvm(String),
    NoTarget,
}

impl From<LLVMString> for Error {
    fn from(err: LLVMString) -> Self {
        Self::Llvm(err.to_string())
    }
}

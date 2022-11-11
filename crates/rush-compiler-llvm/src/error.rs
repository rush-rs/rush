use std::fmt::Display;

use inkwell::support::LLVMString;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Llvm(String),
    NoTarget,
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Llvm(message) => format!("llvm: {message}"),
                Self::NoTarget => "invalid target: no such target".to_string(),
            }
        )
    }
}

impl From<LLVMString> for Error {
    fn from(err: LLVMString) -> Self {
        Self::Llvm(err.to_string())
    }
}

use std::fmt::{self, Display, Formatter};

use inkwell::support::LLVMString;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Llvm(String),
    NoTarget,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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

#[macro_export]
macro_rules! span {
    ($start:literal .. $end:literal) => {
        $crate::Location {
            line: 1,
            column: $start + 1,
            char_idx: $start,
            byte_idx: $start,
            path: "",
        }
        .until($crate::Location {
            line: 1,
            column: $end + 1,
            char_idx: $end,
            byte_idx: $end,
            path: "",
        })
    };
}

#[macro_use]
mod macros;

pub mod ast;
mod error;
mod lexer;
mod parser;
mod span;
mod token;

pub use error::*;
pub use lexer::*;
pub use parser::*;
pub use span::*;
pub use token::*;

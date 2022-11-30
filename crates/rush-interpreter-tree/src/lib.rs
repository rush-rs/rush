mod interpreter;
mod ops;
mod value;

use std::{borrow::Cow, fmt::Debug};

pub use interpreter::*;
use rush_analyzer::Diagnostic;
pub use value::*;

/// Interprets rush source code by walking the analyzed tree.
/// The `Ok(_)` variant returns non-error diagnostics.
/// The `Err(_)` variant returns a `Vec<Diagnostic>` which contains at least one error.
pub fn run<'src>(
    text: &'src str,
    path: &'src str,
) -> Result<Vec<Diagnostic<'src>>, RunError<'src>> {
    let (tree, diagnostics) = rush_analyzer::analyze(text, path)?;
    Interpreter::new().run(tree)?;
    Ok(diagnostics)
}

pub enum RunError<'src> {
    Analyzer(Vec<Diagnostic<'src>>),
    Runtime(Cow<'static, str>),
}

impl<'src> From<Vec<Diagnostic<'src>>> for RunError<'src> {
    fn from(diagnostics: Vec<Diagnostic<'src>>) -> Self {
        Self::Analyzer(diagnostics)
    }
}

impl From<Cow<'static, str>> for RunError<'_> {
    fn from(err: Cow<'static, str>) -> Self {
        Self::Runtime(err)
    }
}

impl Debug for RunError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Analyzer(diagnostics) => write!(f, "{diagnostics:?}"),
            RunError::Runtime(err) => write!(f, "{err:?}"),
        }
    }
}
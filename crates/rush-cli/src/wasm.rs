use std::{fs, path::PathBuf};

use anyhow::{bail, Context};
use rush_analyzer::ast::AnalyzedProgram;
use rush_compiler_wasm::Compiler;

use crate::cli::BuildArgs;

pub fn compile(ast: AnalyzedProgram, args: BuildArgs) -> anyhow::Result<()> {
    let wasm = match Compiler::new().compile(ast) {
        Ok(wasm) => wasm,
        Err(err) => bail!(err),
    };

    // get output path
    let output = match args.output_file {
        Some(out) => out,
        None => {
            let mut path = PathBuf::from(
                args.path
                    .file_stem()
                    .with_context(|| "cannot get filestem of input file")?,
            );
            path.set_extension("wasm");
            path
        }
    };

    fs::write(&output, wasm)
        .with_context(|| format!("cannot write to `{file}`", file = output.to_string_lossy()))?;

    Ok(())
}

use std::{fs, path::PathBuf};

use anyhow::Context;
use rush_analyzer::ast::AnalyzedProgram;
use rush_compiler_x86_64::Compiler;

use crate::cli::BuildArgs;

pub fn compile(ast: AnalyzedProgram, args: BuildArgs) -> anyhow::Result<()> {
    let asm = Compiler::new().compile(ast);

    // get output path
    let output = match args.output_file {
        Some(out) => out,
        None => {
            let mut path = PathBuf::from(
                args.path
                    .file_stem()
                    .with_context(|| "cannot get filestem of input file")?,
            );
            path.set_extension("s");
            path
        }
    };

    fs::write(&output, asm)
        .with_context(|| format!("cannot write to `{file}`", file = output.to_string_lossy()))?;

    Ok(())
}

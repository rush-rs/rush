use std::{
    fs,
    path::PathBuf,
    process::{Command, Stdio},
};

use anyhow::{anyhow, bail, Context};
use rush_analyzer::ast::AnalyzedProgram;
use rush_transpiler_c::Transpiler;
use tempfile::tempdir;

use crate::cli::{BuildArgs, RunArgs};

pub fn compile(ast: AnalyzedProgram, args: BuildArgs) -> anyhow::Result<()> {
    let c = Transpiler::new(false).transpile(ast).to_string();

    // get output path
    let output = match args.output_file {
        Some(out) => out,
        None => {
            let mut path = PathBuf::from(
                args.path
                    .file_stem()
                    .with_context(|| "cannot get filestem of input file")?,
            );
            path.set_extension("c");
            path
        }
    };

    fs::write(&output, c)
        .with_context(|| format!("cannot write to `{file}`", file = output.to_string_lossy()))?;

    Ok(())
}

pub fn run(ast: AnalyzedProgram, args: RunArgs) -> anyhow::Result<i64> {
    let tmpdir = tempdir()?;

    let mut args: BuildArgs = args.try_into()?;

    let c_path = tmpdir.path().join("output.c");
    args.output_file = Some(c_path.clone());
    compile(ast, args)?;

    let bin_path = tmpdir.path().join("output");

    let process = Command::new("gcc")
        .arg(c_path)
        .arg("-std=c89")
        .arg("-o")
        .arg(&bin_path)
        .stderr(Stdio::piped())
        .spawn()?;

    let out = process.wait_with_output()?;
    match out.status.success() {
        true => {}
        false => bail!(
            "compiling C to binary terminated with code {}: {}",
            out.status.code().unwrap_or(1),
            String::from_utf8_lossy(&out.stderr),
        ),
    }

    match Command::new(bin_path).output()?.status.code() {
        Some(code) => Ok(code as i64),
        None => Err(anyhow!("could not get exit-code of rush bin process")),
    }
}

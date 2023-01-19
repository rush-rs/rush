use std::{
    fs,
    path::PathBuf,
    process::{Command, Stdio},
};

use anyhow::{anyhow, bail, Context};
use rush_analyzer::ast::AnalyzedProgram;
use rush_compiler_x86_64::Compiler;
use tempfile::tempdir;

use crate::cli::{BuildArgs, RunArgs};

pub fn compile(ast: AnalyzedProgram, args: BuildArgs) -> anyhow::Result<PathBuf> {
    let asm = Compiler::new().compile(ast);

    // get output path
    let output_path = match args.output_file {
        Some(out) => out,
        None => PathBuf::from(
            args.path
                .file_stem()
                .with_context(|| "cannot get filestem of input file")?,
        ),
    };

    let mut asm_path = output_path.clone();
    asm_path.set_extension("s");
    fs::write(&asm_path, asm).with_context(|| {
        format!(
            "cannot write to `{file}`",
            file = output_path.to_string_lossy()
        )
    })?;

    let tmpdir = tempdir()?;
    let libcore_file_path = tmpdir.path().join("libcore.a");
    let libcore_bytes = include_bytes!("../../rush-compiler-x86-64/corelib/libcore.a");
    fs::write(&libcore_file_path, libcore_bytes)?;

    let process = Command::new("gcc")
        .arg(asm_path)
        .arg(libcore_file_path)
        .arg("-nostdlib")
        .arg("-o")
        .arg(&output_path)
        .stderr(Stdio::piped())
        .spawn()?;

    let out = process.wait_with_output()?;
    match out.status.success() {
        true => {}
        false => bail!(
            "compiling assembly to binary terminated with code {}: {}",
            out.status.code().unwrap_or(1),
            String::from_utf8_lossy(&out.stderr),
        ),
    }

    Ok(output_path)
}

pub fn run(ast: AnalyzedProgram, args: RunArgs) -> anyhow::Result<i64> {
    #[cfg(not(target_arch = "x86_64"))]
    bail!("not running on `x86_64`");

    let args: BuildArgs = args.try_into()?;
    let bin_output = compile(ast, args)?;

    match Command::new(
        bin_output
            .canonicalize()
            .with_context(|| "could not canonicalize output path")?,
    )
    .output()
    .with_context(|| "could not run binary")?
    .status
    .code()
    {
        Some(code) => Ok(code as i64),
        None => Err(anyhow!("could not get exit-code of rush bin process")),
    }
}

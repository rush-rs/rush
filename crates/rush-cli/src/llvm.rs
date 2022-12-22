use std::{
    env, fs,
    process::{self, Command, Stdio},
};

use anyhow::{bail, Context};
use rush_analyzer::ast::AnalyzedProgram;
use rush_compiler_llvm::inkwell::targets::{TargetMachine, TargetTriple};
use tempfile::tempdir;

use crate::cli::{BuildArgs, RunArgs};

pub fn compile(ast: AnalyzedProgram, args: BuildArgs) -> anyhow::Result<()> {
    let target = match args.llvm_target.as_ref() {
        Some(triplet) => TargetTriple::create(triplet),
        None => TargetMachine::get_default_triple(),
    };

    let (obj, ir) = match rush_compiler_llvm::compile(
        ast,
        target,
        args.llvm_opt.into(),
        args.llvm_target.is_none(), // only compile a main fn if target is native
    ) {
        Ok(res) => res,
        Err(err) => bail!(format!("llvm error: {err}")),
    };

    if args.llvm_show_ir {
        println!("{ir}");
    }

    // get output path
    let output = match args.output_file {
        Some(out) => out.with_extension("o"),
        None => args.path.with_extension("o"),
    };

    fs::write(&output, obj.as_slice())
        .with_context(|| format!("cannot write to `{}`", output.to_string_lossy()))?;

    // if a non-native target is used, quit here
    if args.llvm_target.is_some() {
        return Ok(());
    }

    let bin_path = &output.with_extension("");

    dbg!(&bin_path);

    // invoke gcc to link the file
    let command = Command::new("gcc")
        .arg(&output)
        .arg("-o")
        .arg(bin_path)
        .stderr(Stdio::inherit())
        .output()
        .with_context(|| "could not invoke `gcc`")?;

    if !command.status.success() {
        bail!(
            "invoking gcc failed with exit-code {code}",
            code = command.status.code().unwrap()
        )
    }

    Ok(())
}

pub fn run(ast: AnalyzedProgram, args: RunArgs) -> anyhow::Result<i64> {
    let tmpdir = tempdir()?;

    let mut args: BuildArgs = args.try_into()?;

    let executable_path = tmpdir.path().join("output");
    args.output_file = Some(executable_path.clone());

    compile(ast, args)?;

    let command = Command::new(executable_path.to_string_lossy().to_string())
        .stderr(Stdio::inherit())
        .stdout(Stdio::inherit())
        .output()
        .with_context(|| "could not invoke compiled binary")?;

    command
        .status
        .code()
        .map(|c| c as i64)
        .with_context(|| "could not capture exit-code of compiled binary")
}

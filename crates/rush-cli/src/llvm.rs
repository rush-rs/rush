use std::{
    env, fs,
    path::PathBuf,
    process::{self, Command, Stdio},
};

use anyhow::{bail, Context};
use rush_analyzer::ast::AnalyzedProgram;
use rush_compiler_llvm::inkwell::targets::{TargetMachine, TargetTriple};

use crate::cli::{BuildArgs, CompilerBackend, RunArgs};

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

    let out = env::current_dir()
        .with_context(|| "could not determine your working directory")?
        .join({
            let mut base = PathBuf::from(
                args.path
                    .file_stem()
                    .with_context(|| "cannot obtain filestem of output file")?,
            );
            base.set_extension("o");
            base
        });
    let out = out.to_string_lossy();

    // get output path
    let output = match args.output_file {
        Some(out) => out,
        None => args
            .path
            .file_stem()
            .with_context(|| "cannot obtain filestem of output file")?
            .into(),
    };

    // if a non-native target is used, quit here
    if args.llvm_target.is_some() {
        let mut path = output;
        path.set_extension("o");
        fs::write(&path, obj.as_slice())
            .with_context(|| format!("cannot write to `{file}`", file = path.to_string_lossy()))?;
        return Ok(());
    }

    fs::write(out.to_string(), obj.as_slice())
        .with_context(|| format!("cannot write to `{file}`", file = out))?;

    // invoke gcc to link the file
    let command = Command::new("gcc")
        .arg(&out.to_string())
        .arg("-o")
        .arg(output)
        .stderr(Stdio::inherit())
        .output()
        .unwrap_or_else(|err| {
            eprintln!("could not invoke gcc: {err}");
            process::exit(1);
        });

    if !command.status.success() {
        bail!(
            "invoking gcc failed with exit-code {code}",
            code = command.status.code().unwrap()
        )
    }

    Ok(())
}

pub fn run(ast: AnalyzedProgram, args: RunArgs) -> anyhow::Result<i64> {
    let executable = env::current_dir()
        .with_context(|| "could not determine your working directory")?
        .join(
            args.path
                .file_stem()
                .with_context(|| "cannot obtain filestem of output file")?,
        );
    let executable = executable.to_string_lossy();

    compile(
        ast,
        BuildArgs {
            backend: CompilerBackend::Llvm,
            output_file: None,
            llvm_opt: args.llvm_opt,
            llvm_target: None,
            llvm_show_ir: false,
            path: args.path,
        },
    )?;

    let command = Command::new(executable.to_string())
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

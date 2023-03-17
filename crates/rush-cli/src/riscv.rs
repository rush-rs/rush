use std::{
    fs,
    io::ErrorKind,
    path::PathBuf,
    process::{Command, Stdio},
};

use anyhow::{anyhow, bail, Context};
use rush_analyzer::ast::AnalyzedProgram;
use rush_compiler_risc_v::{CommentConfig, Compiler};
use tempfile::TempDir;

use crate::cli::{BuildArgs, RunArgs};

fn command_executable(path: &str) -> anyhow::Result<bool> {
    match Command::new(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
        Err(e) => Err(anyhow!("Unexpected command error: `{e}`")),
    }
}

const GCC_VARIANTS: &[&str] = &["riscv-none-elf-gcc", "riscv64-linux-gnu-gcc"];

pub fn compile(ast: AnalyzedProgram, args: BuildArgs, tmpdir: &TempDir) -> anyhow::Result<()> {
    let asm = Compiler::new().compile(ast, &CommentConfig::Emit { line_width: 32 });

    // get output path
    let output = match args.output_file {
        Some(mut out) => {
            out.set_extension("s");
            out
        }
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

    let mut bin_path = output.clone();
    bin_path.set_extension("");
    let libcore_file_path = tmpdir.path().join("libcore.a");
    let libcore_bytes =
        include_bytes!("../../rush-compiler-risc-v/corelib/libcore-rush-riscv-lp64d.a");
    fs::write(&libcore_file_path, libcore_bytes)?;

    // determine RISC-V GCC
    let mut gcc_bin = None;
    for gcc in GCC_VARIANTS {
        if command_executable(gcc).with_context(|| "Could not determine GCC backend")? {
            gcc_bin = Some(gcc);
            break;
        }
    }
    let Some(gcc) = gcc_bin else {
        bail!(
            "No suitable RISC-V GCC backend has been detected. options: [{}]",
            GCC_VARIANTS.join(", ")
        )
    };

    let process = Command::new(gcc)
        .args(["-nostdlib", "-static"])
        .arg(output)
        .arg(libcore_file_path)
        .arg("-o")
        .arg(&bin_path)
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("could not invoke `{gcc}`"))?;

    let out = process.wait_with_output()?;
    match out.status.success() {
        true => {}
        false => bail!(
            "compiling assembly to binary terminated with code {}: {}",
            out.status.code().unwrap_or(1),
            String::from_utf8_lossy(&out.stderr),
        ),
    }

    Ok(())
}

pub fn run(ast: AnalyzedProgram, args: RunArgs) -> anyhow::Result<i64> {
    let tmpdir = tempfile::tempdir()?;

    let mut args: BuildArgs = args.try_into()?;
    let asm_path = tmpdir.path().join("output.s");
    let bin_path = tmpdir.path().join("output");
    args.output_file = Some(asm_path);

    compile(ast, args, &tmpdir)?;

    match Command::new("qemu-riscv64")
        .arg(bin_path)
        .output()
        .with_context(|| "could not invoke `qemu-riscv64`")?
        .status
        .code()
    {
        Some(code) => Ok(code as i64),
        None => Err(anyhow!("could not get exit-code of rush bin process")),
    }
}

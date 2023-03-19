use std::fs;

use anyhow::{bail, Context};
use rush_analyzer::ast::AnalyzedProgram;
use rush_interpreter_vm::Vm;

use crate::cli::{BuildArgs, RunArgs};

const VM_MEM_SIZE: usize = 10_024;

pub fn compile(ast: AnalyzedProgram, args: BuildArgs) -> anyhow::Result<()> {
    let program = rush_interpreter_vm::compile(ast);

    // get output path
    let output = match args.output_file {
        Some(out) => out,
        None => args.path.with_extension("s"),
    };

    fs::write(&output, program.to_string())
        .with_context(|| format!("cannot write to `{}`", output.to_string_lossy()))?;

    Ok(())
}

pub fn run(ast: AnalyzedProgram, args: RunArgs) -> anyhow::Result<i64> {
    let mut vm: Vm<VM_MEM_SIZE> = Vm::new();
    let program = rush_interpreter_vm::compile(ast);

    match args.vm_speed {
        Some(clock) => {
            if clock == 0 {
                bail!("attempted to use a clock speed of 0, must be > 0")
            }
            vm.debug_run(program, clock)
        }
        None => vm.run(program),
    }
    .with_context(|| "VM crashed")
}

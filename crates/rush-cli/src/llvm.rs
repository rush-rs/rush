use std::{
    env, fs,
    path::Path,
    process::{self, Command, Stdio},
};

use rush_analyzer::ast::AnalyzedProgram;
use rush_compiler_llvm::inkwell::targets::{TargetMachine, TargetTriple};

use crate::cli::BuildArgs;

pub fn compile(ast: AnalyzedProgram, args: BuildArgs) {
    let (obj, ir) = rush_compiler_llvm::compile(
        ast,
        match args.llvm_target {
            Some(triplet) => TargetTriple::create(&triplet),
            None => TargetMachine::get_default_triple(),
        },
        args.llvm_opt.into(),
    )
    .unwrap_or_else(|err| {
        eprintln!("compilation failed: llvm error: {err}");
        process::exit(1);
    });

    if args.llvm_show_ir {
        println!("{ir}");
    }
    // write to file
    let out = Path::new("/tmp").join("rush.o");

    fs::write(&out, obj.as_slice())
        .unwrap_or_else(|err| eprintln!("cannot write to `{}`: {err}", out.to_string_lossy()));

    // get output path
    let output = args.output_file.unwrap_or_else(|| {
        args.path
            .file_stem()
            .expect("file reading would have failed before")
            .into()
    });

    // invoke gcc to link the file
    let command = Command::new("gcc")
        .arg(&out)
        .arg("-o")
        .arg(output)
        .stderr(Stdio::inherit())
        .output()
        .unwrap_or_else(|err| {
            eprintln!("could not invoke gcc: {err}");
            process::exit(1);
        });

    if !command.status.success() {
        eprintln!(
            "gcc failed with exit-code {}",
            command.status.code().unwrap()
        );
        process::exit(1);
    }
}

pub fn run(ast: AnalyzedProgram, args: BuildArgs) {
    // get executable path
    let executable: String = match &args.output_file {
        Some(path) => match path.is_absolute() {
            true => path.to_string_lossy().into(),
            false => env::current_dir()
                .expect("there must be a working dir")
                .join(path)
                .to_string_lossy()
                .into(),
        },
        None => env::current_dir()
            .expect("there must be a working dir")
            .join(
                args.path
                    .file_stem()
                    .expect("file reading would have failed before"),
            )
            .to_string_lossy()
            .into(),
    };

    compile(ast, args);

    let command = Command::new(executable)
        .stderr(Stdio::inherit())
        .stdout(Stdio::inherit())
        .output()
        .unwrap_or_else(|err| {
            eprintln!("could not invoke binary: {err}");
            process::exit(1);
        });
    process::exit(command.status.code().unwrap_or_default())
}

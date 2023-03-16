use std::{
    io::ErrorKind,
    path::Path,
    process::{Command, Stdio},
};

fn command_executable(path: &str) -> bool {
    match Command::new(path).spawn() {
        Ok(_) => true,
        Err(e) if e.kind() == ErrorKind::NotFound => false,
        Err(e) => panic!("Unknown error: `{e}`"),
    }
}

const ASSEMBLERS: &[&str] = &["riscv64-alpine-linux-musl-as", "riscv64-linux-gnu-as"];
const ARCHIVERS: &[&str] = &["riscv64-alpine-linux-musl-ar", "riscv64-linux-gnu-ar"];
const GCC_VARIANTS: &[&str] = &["riscv-none-elf-gcc", "riscv64-linux-gnu-gcc"];

fn main() {
    //// X86_64 corelib ////
    // rebuild if the corelib directory has changed
    println!("cargo:rerun-if-changed=../rush-compiler-x86-64/corelib/src");
    println!("cargo:rerun-if-changed=../rush-compiler-x86-64/corelib/Makefile");

    let path = Path::canonicalize(Path::new("../rush-compiler-x86-64/corelib/")).unwrap();
    let out = Command::new("make")
        .arg("libcore.a")
        .current_dir(path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();

    match out.status.success() {
        true => {}
        false => panic!(
            "building x86 `libcore.a` terminated with code {}: {}",
            out.status.code().unwrap(),
            String::from_utf8_lossy(&out.stderr)
        ),
    }

    //// RISC-V corelib ////
    // rebuild if the corelib directory has changed
    println!("cargo:rerun-if-changed=../rush-compiler-risc-v/corelib/src");
    println!("cargo:rerun-if-changed=../rush-compiler-x86-64/corelib/Makefile");

    // determine RISC-V assembler
    let mut assembler_bin = None;
    for assembler in ASSEMBLERS {
        if command_executable(assembler) {
            assembler_bin = Some(assembler);
            break;
        }
    }

    // determine RISC-V archiver
    let mut archiver_bin = None;
    for archiver in ARCHIVERS {
        if command_executable(archiver) {
            archiver_bin = Some(archiver);
            break;
        }
    }

    // determine RISC-V GCC
    let mut gcc_bin = None;
    for gcc in GCC_VARIANTS {
        if command_executable(gcc) {
            gcc_bin = Some(gcc);
            break;
        }
    }

    let path = Path::canonicalize(Path::new("../rush-compiler-risc-v/corelib/")).unwrap();
    let out = Command::new("make")
        .arg("all")
        .arg(format!(
            "assembler={}",
            assembler_bin.expect("No suitable RISC-V assembler detected")
        ))
        .arg(format!(
            "archiver={}",
            archiver_bin.expect("No suitable RISC-V archiver detected")
        ))
        .arg(format!(
            "gcc={}",
            gcc_bin.expect("No suitable RISC-V gcc detected")
        ))
        .current_dir(path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();

    match out.status.success() {
        true => {}
        false => panic!(
            "building RISC-V `libcore.a` terminated with code {}: {}",
            out.status.code().unwrap(),
            String::from_utf8_lossy(&out.stderr)
        ),
    }
}

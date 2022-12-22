use std::{
    path::Path,
    process::{Command, Stdio},
};

fn main() {
    //// X86_64 corelib ////
    // rebuild if the corelib directory has changed
    println!("cargo:rerun-if-changed=../rush-compiler-x86-64/corelib/");

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
    println!("cargo:rerun-if-changed=../rush-compiler-risc-v/corelib/");

    let path = Path::canonicalize(Path::new("../rush-compiler-risc-v/corelib/")).unwrap();
    let out = Command::new("make")
        .arg("all")
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

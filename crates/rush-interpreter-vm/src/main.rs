use std::{env, fs, process};

fn main() {
    let path = env::args().nth(1).unwrap();
    let debug = matches!(env::args().nth(2), Some(arg) if arg == "debug");

    let clock = if debug {
        env::args().nth(3).unwrap().parse().unwrap()
    } else {
        0
    };

    let code = fs::read_to_string(&path).unwrap();
    let (ast, _) = rush_analyzer::analyze(&code, &path).unwrap();

    let out = rush_interpreter_vm::compile(&ast).to_string();
    println!("{out}");
    fs::write("output.s", out).unwrap();

    let code = match debug {
        true => rush_interpreter_vm::debug_run(&ast, clock),
        false => rush_interpreter_vm::run(&ast),
    }
    .unwrap_or_else(|err| {
        eprintln!("\x1b[1;31mVM crashed\x1b[1;0m: {} -> {}", err.kind, err.msg);
        process::exit(1);
    });
    println!("Program exited with code {code}");
    process::exit(code as i32);
}

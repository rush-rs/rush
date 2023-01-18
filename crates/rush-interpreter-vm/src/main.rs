use std::{env, fs, process, time::Instant};

use rush_interpreter_vm::Vm;

fn main() {
    let path = env::args().nth(1).unwrap();
    let debug = matches!(env::args().nth(2), Some(arg) if arg == "debug");

    let clock = if debug {
        env::args().nth(3).unwrap().parse().unwrap()
    } else {
        0
    };

    let code = fs::read_to_string(&path).unwrap();

    let mut start = Instant::now();

    let (ast, _) = rush_analyzer::analyze(&code, &path).unwrap();

    let program = rush_interpreter_vm::compile(ast);

    if debug {
        println!("{program}");
    }

    println!("compilation: {:?}", start.elapsed());

    start = Instant::now();

    let code = match debug {
        true => Vm::new().debug_run(program, clock),
        false => Vm::new().run(program),
    }
    .unwrap_or_else(|err| {
        eprintln!("\x1b[1;31mVM crashed\x1b[1;0m: {} -> {}", err.kind, err.msg);
        process::exit(1);
    });
    println!("Program exited with code {code}");
    println!("execution: {:?}", start.elapsed());
    process::exit(code as i32);
}

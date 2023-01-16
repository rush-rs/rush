use std::{env, fs, process, time::Instant};

use rush_interpreter_vm::Vm;

const MEM_SIZE: usize = 500;

fn main() {
    let path = env::args().nth(1).unwrap();
    let debug = matches!(env::args().nth(2), Some(arg) if arg == "debug");

    let clock = if debug {
        env::args().nth(3).unwrap().parse().unwrap()
    } else {
        0
    };

    let start = Instant::now();

    let code = fs::read_to_string(&path).unwrap();
    let (ast, _) = rush_analyzer::analyze(&code, &path).unwrap();

    let program = rush_interpreter_vm::compile(ast);

    if debug {
        println!("{program}");
    }

    let mut vm: Vm<MEM_SIZE> = Vm::new();

    let code = match debug {
        true => vm.debug_run(program, clock),
        false => vm.run(program),
    }
    .unwrap_or_else(|err| {
        eprintln!("\x1b[1;31mVM crashed\x1b[1;0m: {} -> {}", err.kind, err.msg);
        process::exit(1);
    });
    println!("Program exited with code {code}");
    println!("{:?}", start.elapsed());
    process::exit(code as i32);
}

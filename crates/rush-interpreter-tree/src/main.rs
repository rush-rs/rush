use std::{env, fs, process, time::Instant};

fn main() {
    let start = Instant::now();
    let path = env::args().nth(1).unwrap();
    let code = fs::read_to_string(&path).unwrap();
    let exit_code = rush_interpreter_tree::run(&code, &path).unwrap().0;
    println!("Program exited with code {exit_code}");
    println!("{:?}", start.elapsed());
    process::exit(exit_code as i32);
}

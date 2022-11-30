use std::{env, fs, time::Instant};

fn main() {
    let start = Instant::now();
    let path = env::args().nth(1).unwrap();
    let code = fs::read_to_string(&path).unwrap();
    rush_interpreter_tree::run(&code, &path).unwrap();
    println!("{:?}", start.elapsed());
}

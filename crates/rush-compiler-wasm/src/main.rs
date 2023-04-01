use std::{env, fs, time::Instant};

fn main() {
    let start = Instant::now();
    let path = env::args().nth(1).unwrap();
    let code = fs::read_to_string(&path).unwrap();
    let bytes = rush_compiler_wasm::compile(&code, &path)
        .unwrap()
        .unwrap()
        .0;
    fs::write("output.wasm", bytes).unwrap();
    println!("{:?}", start.elapsed());
}

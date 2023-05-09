use std::{env, fs, time::Instant};

use rush_compiler_x86_64::CommentConfig;

fn main() {
    let args = env::args().collect::<Vec<_>>();

    let comment_config = match args.get(2).unwrap_or(&String::from("65")).parse::<usize>() {
        Ok(0) | Err(_) => CommentConfig::NoComments,
        Ok(line_width) => CommentConfig::Emit { line_width },
    };

    let start = Instant::now();
    let path = &args[1];
    let code = fs::read_to_string(path).unwrap();
    let bytes = rush_compiler_x86_64::compile(&code, path, comment_config)
        .unwrap()
        .0;
    fs::write("output.s", bytes).unwrap();
    println!("{:?}", start.elapsed());
}

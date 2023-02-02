use std::{env, fs, process, time::Instant};

use rush_compiler_risc_v::CommentConfig;

fn main() {
    let path = env::args().nth(1).unwrap();
    let line_width = env::args().nth(2).unwrap_or("32".to_string()).parse().unwrap();
    let code = fs::read_to_string(&path).unwrap();
    let start = Instant::now();
    let (out, diagnostics) = rush_compiler_risc_v::compile(
        &code,
        &path,
        &CommentConfig::Emit {
            line_width,
        },
    )
    .unwrap_or_else(|diagnostics| {
        println!(
            "{}",
            diagnostics
                .iter()
                .map(|d| format!("{d:#}"))
                .collect::<Vec<String>>()
                .join("\n\n")
        );
        process::exit(1)
    });

    println!(
        "{}",
        diagnostics
            .iter()
            .map(|d| format!("{d:#}"))
            .collect::<Vec<String>>()
            .join("\n\n")
    );

    println!("compile: {:?}", start.elapsed());
    fs::write("output.s", out).unwrap();
}

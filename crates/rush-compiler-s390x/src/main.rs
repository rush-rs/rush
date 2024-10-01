use std::{env, fs, process, time::Instant};

use rush_compiler_s390x::CommentConfig;

fn main() {
    let path = env::args().nth(1).unwrap();

    let line_width = env::args()
        .nth(2)
        .unwrap_or("32".to_string())
        .parse()
        .unwrap();

    let code = fs::read_to_string(&path).unwrap();
    let start = Instant::now();

    let comment_config = match env::args().nth(3) {
        Some(input) if input == *"n" => CommentConfig::NoComments,
        Some(other) => panic!("illegal comment config: {other}"),
        None => CommentConfig::Emit { line_width },
    };

    let (out, diagnostics) = rush_compiler_s390x::compile(&code, &path, &comment_config)
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

    if Some("-t".to_string()) == env::args().nth(3) {
        println!("compile: {:?}", start.elapsed());
    }
    fs::write("output.s", out).unwrap();
}

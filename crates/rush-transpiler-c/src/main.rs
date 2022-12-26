use std::{env, fs, process, time::Instant};

fn main() {
    let path = env::args().nth(1).unwrap();
    let emit_comments = env::args().nth(2) == Some("true".to_string());
    let code = fs::read_to_string(&path).unwrap();
    let start = Instant::now();
    let (out, diagnostics) = rush_transpiler_c::transpile(&code, &path, emit_comments)
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

    println!("tanspile: {:?}", start.elapsed());
    fs::write("output.c", out).unwrap();
}

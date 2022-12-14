use std::{env, fs, process, time::Instant};

fn main() {
    let path = env::args().nth(1).unwrap();
    let code = fs::read_to_string(&path).unwrap();
    let start = Instant::now();
    let (out, diagnostics) =
        rush_compiler_risc_v::compile(&code, &path).unwrap_or_else(|diagnostics| {
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

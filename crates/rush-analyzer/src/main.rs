use std::{env, fs, process, time::Instant};

fn main() {
    let start = Instant::now();
    let path = env::args().nth(1).unwrap();
    let code = fs::read_to_string(&path).unwrap();
    match rush_analyzer::analyze(&code, &path) {
        Ok(res) => {
            for diagnostic in &res.1 {
                println!("{diagnostic:#}");
            }
            res.0
        }
        Err(diagnostics) => {
            for diagnostic in diagnostics {
                println!("{diagnostic:#}");
            }
            eprintln!("Analyzer detected issues");
            process::exit(1);
        }
    };
    if Some("-t".to_string()) == env::args().nth(2) {
        println!("{:?}", start.elapsed());
    }
}

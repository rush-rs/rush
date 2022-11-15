use clap::Parser;
use cli::{Args, Command};

mod cli;

fn main() {
    let args = Args::parse();
    match args.command {
        Command::Build => todo!("Compilation"),
        Command::Run => todo!("Run"),
        Command::Check => todo!("Check"),
    }
}

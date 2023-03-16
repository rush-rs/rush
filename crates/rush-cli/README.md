# Command Line Interface

This tool combines all of rush's backends and packages them in an easy-to-use
CLI.

## Prerequisites

- The [Rust](https://www.rust-lang.org/tools/install) programming language (with
  Cargo)
- The dependencies of the
  [x86_64 Compiler](https://github.com/rush-rs/rush/tree/main/crates/rush-compiler-x86-64)
- The dependencies of the
  [RISC-V Compiler](https://github.com/rush-rs/rush/tree/main/crates/rush-compiler-risc-v)
- GNU Make

### A Note on LLVM

- Since the LLVM compiler backend requires a valid LLVM installation, the `llvm`
  feature is disabled by default.

## Installation

- Clone the rush repository
- Navigate to the CLI directory
- Install the CLI

```bash
git clone https://github.com/rush-rs/rush && \
cd rush/crates/rush-cli && \
cargo install --path=.
```

### With the Additional LLVM Feature

> **Note:** A valid LLVM installation is required. For detailed instructions,
> please refer to the [Inkwell](https://github.com/TheDan64/inkwell) crate.

```bash
git clone https://github.com/rush-rs/rush && \
cd rush/crates/rush-cli && \
cargo install --path=. -F llvm
```

## Usage

- Prerequisite: A file ending in `.rush` and a valid CLI installation
- After installation, the `rush-cli` command should be available

### Compilation

- If the command is executed, a list of possible backends is displayed
- Any of these backends can be set
- An optional output name can be set using the `-o filename` flag

```bash
rush-cli build fib.rush --backend=your-backend
```

#### Example: Compilation using RISC-V / x86_64

```bash
# RISC-V:
rush-cli build fib.rush --backend=risc-v -o my_output
# x86_64:
rush-cli build fib.rush --backend=x86-64 -o my_output
```

- Now, there is a file named `my_output` and one named `my_output.s`
- The former is an executable binary, the latter is the generated assembly file

### Execution

- If the command below is executed, a list of available run-backends is
  displayed
- Any of these backends may be used in order to run the rush program

```bash
rush-cli run fib.rush --backend=your-backend
```

#### Example: Running a rush Program

```bash
# RISC-V:
rush-cli run fib.rush --backend=risc-v
# VM
rush-cli run fib.rush --backend=vm
# Tree-walking interpreter
rush-cli run fib.rush --backend=tree
```

- All of these run-commands will omit any output files as temporary directories
  are used for the build artifacts

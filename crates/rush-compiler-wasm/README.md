# WebAssembly Compiler Backend

A single-target rush compiler backend which generates
[Wasm](https://webassembly.org/) files without the need for external
dependencies.

## Prerequisites

Since the compiler itself requires no external dependencies, only a WebAssembly
runtime is to be installed. Hence, program execution requires a
[Wasm runtime](https://wasmer.io/).

### Runtime

On Arch-Linux based systems, the following package can be installed to set up a
working Wasm runtime.

- `wasmer` for executing `.wasm` files

## Compiling and Running rush Code

### Compilation of rush Code

- Prerequisite: A file ending in `.rush` which contains the program.
- Execute following command as it will generate the `output.wasm` file from the
  source program.

```bash
cargo run your-program.rush
```

### Running Wasm Files

- Since you have installed a Wasm runtime prior to reading this section,
  following command can be used to execute the `.wasm` file using this runtime.

```bash
wasmer output.wasm
```

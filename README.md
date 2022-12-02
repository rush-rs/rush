# The rush Programming Language

A simple programming language for researching different ways of program
execution and compilation.

> **Note:** Since this project is part of a research project, the language
> cannot be considered _production-ready_.

## Projects Which Are Part of rush

### Program Analysis

- [Lexer & Parser](./crates/rush-parser/)
- [Semantic Analyzer](./crates/rush-analyzer/)

### Interpreters

- [Tree-walking Interpreter](./crates/rush-interpreter-tree/)
- [Virtual Machine Interpreter](./crates/rush-interpreter-vm/)

### Compilers

- [WASM Compiler](./crates/rush-compiler-wasm/)
- [LLVM Compiler](./crates/rush-compiler-llvm/)
- [RISC-V Compiler](./crates/rush-compiler-risc-v/)
- [x86_64 Compiler](./crates/rush-compiler-x86-64/)

### Toolchain

- [Language Server](./crates/rush-ls/)
- [rush CLI](./crates/rush-cli/)

> **Note:** This project is in early active development

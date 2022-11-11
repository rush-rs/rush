# The rush Programming Language

A simple programming language for researching different ways of program
execution and compilation.

## Semantics

Each rush program consists of an arbitrary amount of functions. In order to
create a valid program, the `main` function needs to be declared.

### Functions

In rush, there cannot be top-level code other than function declarations.
Therefore, a [main](#the-main-function) function is required to mark the start
of program execution.

### The main function

The `main` function serves as the entry to a rush program because program
execution will start there. This concept is very similar to the `main` function
in [Rust](https://www.rust-lang.org/) or in
[C](https://en.wikipedia.org/wiki/C_(programming_language)). The function
signature of the `main` function has to look like this:

```rs
fn main() {
  ...
}
```

This means the `main` function cannot take any `arguments` or return a value.

## Types

## Operators

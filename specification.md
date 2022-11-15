# The rush Programming Language

A simple programming language for researching different ways of program
execution and compilation.

## Semantics

Each rush program consists of an arbitrary amount of functions. In order to
create a valid program, the `main` function needs to be declared.

### Functions

In rush, there cannot be top-level code other than function declarations and
globals. Therefore, a [main](#the-main-function) function is required to mark
the start of program execution.

### The main function

The `main` function serves as the entry to a rush program because program
execution will start there. This concept is very similar to the `main` function
in [Rust](https://www.rust-lang.org/) or in
[C](https://en.wikipedia.org/wiki/C_(programming_language)). The function
signature of the `main` function has to look like this:

```rs
fn main() {
    // ...
}
```

This means the `main` function cannot take any `arguments` or return a non-unit
value.

## Types

| Notation    | Example Value | Size   | Values                          |
| ----------- | ------------- | ------ | ------------------------------- |
| `int`       | 42            | 64 bit | $- (2 ^{31})$ to $2 ^ {31}$     |
| `float`     | 3.1415        | 64 bit | $- (2.0 ^{31})$ to $2.0 ^ {31}$ |
| `char`      | 'a'           | 8 bit  | $0$ to $127$                    |
| `bool`      | true          | 1 bit  | `true` and `false`              |
| `()` (unit) | no value      | 1 bit  | no values                       |

## Operators

### Arithmetic Operators

| Operator | Operand Types  | Produces (Type)  |
| -------- | -------------- | ---------------- |
| +        | `int`, `float` | same as operands |
| -        | `int`, `float` | same as operands |
| *        | `int`, `float` | same as operands |
| /        | `int`, `float` | same as operands |
| %        | `int`          | `int`            |
| **       | `int`          | `int`            |

### Bitwise Operators

| Operator | Operand Types | Produces (Type)  |
| -------- | ------------- | ---------------- |
| <<       | `int`         | `int`            |
| >>       | `int`         | `int`            |
| \|       | `int`, `bool` | same as operands |
| \&       | `int`, `bool` | same as operands |
| \^       | `int`, `bool` | same as operands |

### Logical Operators

| Operator | Operand Types                  | Produces (Type) |
| -------- | ------------------------------ | --------------- |
| &&       | `bool`                         | `bool`          |
| \|\|     | `bool`                         | `bool`          |
| <        | `int`, `float`                 | `bool`          |
| <=       | `int`, `float`                 | `bool`          |
| >        | `int`, `float`                 | `bool`          |
| >=       | `int`, `float`                 | `bool`          |
| ==       | `int`, `float`, `bool`, `char` | `bool`          |
| !=       | `int`, `float`, `bool`, `char` | `bool`          |

## Type Cast

The rush language supports conversion between types. The basic syntax looks like
this:

```rs
type as type
```

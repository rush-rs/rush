# Semantic Language Specification

For an overview about rush's syntax, please consult the
[grammar](./grammar.ebnf).

## Semantics

Each rush program consists of an arbitrary amount of functions and global
variables. In order to create a valid program, the [`main`](#the-main-function)
function needs to be declared.

### Functions

In rush, there cannot be top-level code other than function declarations and
globals.

#### The Main Function

The `main` function serves as the entry to a rush program, so that code starts
executing from here. This concept is very similar to the `main` function in
[Rust](https://www.rust-lang.org/) or in
[C](https://en.wikipedia.org/wiki/C_(programming_language)). The function
signature of the `main` function has to look like this:

```rs
fn main() {
    // ...
}
```

Therefore the `main` function cannot take any arguments or return a non-unit
type value.

#### Builtin Functions

##### Exit

```rs
fn exit(code: i32) -> !
```

The `exit` function calls the operating system, demanding to quit the program
with the specified exit-code.

## Types

| Notation    | Example Value | Size   | Values                         |
| ----------- | ------------- | ------ | ------------------------------ |
| `int`       | 42            | 64 bit | $- 2 ^{63} \le x \lt 2 ^ {63}$ |
| `float`     | 3.1415        | 64 bit | IEEE float values              |
| `char`      | 'a'           | 8 bit  | $0 \le x \le 127$              |
| `bool`      | true          | 1 bit  | `true` and `false`             |
| `()` (unit) | no value      | 1 bit  | no values                      |

## Prefix Operators

| Operator | Operand Type          | Produces (Type) |
| -------- | --------------------- | --------------- |
| -        | `int`                 | `int`           |
| !        | `bool`, `int`, `char` | same as operand |

## Infix Operators

### Arithmetic Operators

| Operator | Operand Types  | Produces (Type)  |
| -------- | -------------- | ---------------- |
| +        | `int`, `float` | same as operands |
| -        | `int`, `float` | same as operands |
| *        | `int`, `float` | same as operands |
| /        | `int`, `float` | same as operands |
| %        | `int`          | `int`            |
| **       | `int`          | `int`            |

> **Note:** Division by zero using `/` or `%` is undefined behavior and may vary
> per backend.

### Bitwise Operators

| Operator | Operand Types | Produces (Type)  |
| -------- | ------------- | ---------------- |
| <<       | `int`         | `int`            |
| >>       | `int`         | `int`            |
| \|       | `int`, `bool` | same as operands |
| \&       | `int`, `bool` | same as operands |
| \^       | `int`, `bool` | same as operands |

> **Note:** Shifting by a number outside the range `0..=63` is undefined
> behavior and may vary per backend.

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

> **Note:** All logical infix operators require values of equal types on their
> left- and right-hand sides.

## Type Cast

The rush language supports conversion between types. The basic syntax looks like
this:

```rs
value as type
```

### Valid Casts

| From    | To      | Notes                                      |
| ------- | ------- | ------------------------------------------ |
| `int`   | `int`   | redundant                                  |
| `int`   | `float` |                                            |
| `int`   | `bool`  | `res` = `int` != 0                         |
| `int`   | `char`  | [defined here](#a-note-on-casting-to-char) |
| `float` | `int`   | truncate                                   |
| `float` | `float` | redundant                                  |
| `float` | `bool`  | `res` = `float` != 0.0                     |
| `float` | `char`  | [defined here](#a-note-on-casting-to-char) |
| `bool`  | `bool`  | redundant                                  |
| `bool`  | `int`   | `true` = 1 \| `false` = 0                  |
| `bool`  | `float` |                                            |
| `bool`  | `char`  |                                            |
| `char`  | `char`  | redundant                                  |
| `char`  | `int`   |                                            |
| `char`  | `float` |                                            |
| `char`  | `bool`  | `res` = `int(char)` != 0                   |

### A Note on Casting to Char

When casting `int` or `float` values to char, any source value $lt 0$ is
transformed into a `char` with the value $0$. Furthermore, if the source value
is $\gt127$, the resulting char will have the value $127$. These limitations are
due to chars only containing valid `ASCII` characters which lie in the range $0
\le x \le 127$.

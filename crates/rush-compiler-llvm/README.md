# LLVM Compiler Backend

A multiple-target rush compiler backend using the _LLVM_ framework and the
[Inkwell](https://github.com/TheDan64/inkwell) crate.

## Prerequisites

In order to compile the _LLVM_ backend, please ensure that you have a valid
_LLVM_ installation on your machine. For further information and
troubleshooting, please consult the
[llvm-sys](https://docs.rs/crate/llvm-sys/latest) crate's documentation as it is
used by [Inkwell](https://github.com/TheDan64/inkwell).

## Compiling and Running rush Code

### Compilation

- Prerequisite: A file ending in `.rush` which contains the program.
- Execute following command as it will generate the `output.o` file from the
  source program.

```bash
cargo run your-file.rush
```

### Linking

- Since rush programs depend on some sort of standard library, linking the `.o`
  file using `gcc` will likely result in a valid program.
- In order to link the `output.o` alongside your system's C library, issue
  following command.

```bash
gcc output.o -o output
```

### Running the Program

- Since the output of the previous command is a binary, it can be executed using
  the following command.
- The suffix containing `echo ...` is optional and merely prints out the
  program's exit-code.

```bash
./output ; echo $?
```

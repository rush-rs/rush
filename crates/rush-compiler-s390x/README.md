# S390x Compiler Backend

A single-target rush compiler backend which generates
[IBM S390x](https://en.wikipedia.org/wiki/IBM_System/390) assembly files without the need for heavy external
dependencies.

## Prerequisites

Since the compiler itself requires no external dependencies, only the S390X
toolchain and additional software is to be installed. Hence, program execution
requires an assembler, a linker, and an emulator.

### Toolchain

On NixOS, the following command is to be executed to obtain a shell with a cross compiler targeting S390X.

```bash
nix-shell ./crossShell.nix
```

### Emulator

On NixOS, the `qemu_full` package provides an
emulator for S390X processors.

## Compiling and Running rush Code

### Compilation of rush Code

- Prerequisite: A file ending in `.rush` which contains the program.
- Execute following command as it will generate the `output.s` file from the
  source program.

```bash
cargo run your-program.rush
```

Since S390X targeted rush programs depend on a special [corelib](./corelib),
linking demands more steps than usual.

### Assembling the Corelib

- Navigate inside the [corelib](./corelib/) subdirectory.
- Enter the following command in order to compile the corelib for several RISC-V
  ABIs as it should execute successfully and produce several files ending in
  `.a`

```bash
make all
```

### Final Assembling & Linking Alongside the Corelib

- This project includes a `Makefile` which contains the `build` target.
- Issuing the following command should produce an executable binary file in the
  current directory.

```bash
make build
```

### Running S390X Binaries

- Since you have installed a S390X emulator prior to reading this section,
  following command can be used to run the binary using the emulator.
- The suffix containing `echo ...` is optional and merely prints out the
  program's exit-code.

```bash
qemu-390x ./your-output ; echo $?
```

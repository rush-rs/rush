# RISC-V Compiler Backend

A single-target rush compiler backend which generates
[RISC-V](https://riscv.org/) assembly files without the need for external
dependencies.

## Prerequisites

Since the compiler itself requires no external dependencies, only the RISC-V
toolchain and additional software is to be installed. Hence, program execution
requires an assembler, a linker, and an emulator.

### Toolchain

On Arch-Linux based systems, the following packages can be installed to set up a
working toolchain.

- `riscv64-linux-gnu-gcc` for assembling and linking
- `riscv64-linux-gnu-binutils` (optional)

### Emulator

On Arch-Linux based systems, the `qemu-system-riscv` package provides an
emulator for RISC-V processors.

## Compiling and Running rush Code

### Compilation of rush Code

- Prerequisite: A file ending in `.rush` which contains the program.
- Execute following command as it will generate the `output.s` file from the
  source program.

```bash
cargo run your-program.rush
```

Since RISC-V targeted rush programs depend on a special [corelib](./corelib),
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

### Running RISC-V Binaries

- Since you have installed a RISC-V emulator prior to reading this section,
  following command can be used to run the binary using the emulator.
- The suffix containing `echo ...` is optional and merely prints out the
  program's exit-code.

```bash
qemu-riscv64 ./your-output ; echo $?
```

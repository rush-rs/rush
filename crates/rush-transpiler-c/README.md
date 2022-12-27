# C Transpiler Backend

A rush transpiler which generates
[ANSI C](https://en.wikipedia.org/wiki/C_(programming_language)#ANSI_C_and_ISO_C)
code without the need for external dependencies.

## Running rush Code

- Prerequisite: A file ending in `.rush` which contains the program.
- Execute the following command in order to run the program.

```bash
cargo run your-file.rush
```

- Execute the following command in order to compile and then run the emitted `C`
  file
- The `-std=c89` flag is completely optional but shows that the transpiler
  indeed generates `ANSI C`.

```bash
gcc output.c -std=c89 -o out && ./out
```

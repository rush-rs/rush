# VM Interpreter Backend

A rush Interpreter using a micro-compiler alongside a VM for its runtime.

## Running rush Code

- Prerequisite: A file ending in `.rush` which contains the program.
- Execute the following command in order to run the program.

```bash
cargo run your-file.rush
```

### Debugging Output

- When debugging output is desired, following command is to be used.
- The `1` at the end specifies how many instructions per second are executed by
  the VM.
- In this case, the VM will operate at its minimum speed.
- Hence, a larger value will result in faster program execution

```bash
cargo run your-file.rush debug 1
```

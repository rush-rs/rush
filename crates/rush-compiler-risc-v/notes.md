# RISC-V Notes

## Stack Layout

In the current implementation, the stack pointer `sp` includes legal memory
locations whilst the frame pointer `fp` points to the address after the last
legal one.

```
                      0(sp)   8(sp)  16(sp)  24(sp)      32(sp)
                    -32(fp) -24(fp) -16(fp)  -8(fp)       0(fp)
                       |       |        |        |         |
                  +--------+--------+--------+--------+
                  |        |        |        |        |
illegal memory <- |   fp   |   ra   |   ax   |   ax   | -> illegal memory
                  |        |        |        |        |
                  +--------+--------+--------+--------+
                      ^                                    ^
                      |                                    |
                      sp                                   fp
```

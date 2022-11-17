src := pow exit cast
obj := pow.o exit.o cast.o
dest := libcore.a

# creates an `.a` file of the entire corelib
corelib:
	rm -f $(dest)
	for file in $(src); do \
		riscv64-linux-gnu-as src/$$file.s -o $$file.o; \
	done

	riscv64-linux-gnu-ar -cq $(dest) $(obj)
	rm $(obj)

# runs `main.c` alongside the corelib in order to test it
test: corelib
	riscv64-linux-gnu-gcc -nostdlib -static main.c -L . -lcore -o main
	qemu-riscv64 ./main

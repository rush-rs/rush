src := ../../samples/main.rush
out := test

ibm:
	cargo r -- ./fib.rush
	( echo ".globl main"; cargo r -- ./fib.rush s390x-unknown-linux-gnu 2> /dev/null; sed -e 's/\..*//g' output.s | grep '\S' ) > output2.s
	bat output2.s --language=asm

run: $(src) src
	cargo r $(src)
	gcc output.o -o $(out)
	./$(out)

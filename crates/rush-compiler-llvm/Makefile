src := ../../samples/test.rush
out := test

run: $(src) src
	cargo r $(src)
	gcc output.o -o $(out)
	./$(out)

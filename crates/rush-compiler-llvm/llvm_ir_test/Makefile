run: build
	./main

# builds the main.cpp file
build:
	clang main.cpp -o main

# generates the IR from the main.cpp file
ir:
	clang -S -emit-llvm main.cpp -o main.ll

# builds an executable from the IR file
build-ir:
	llc main.ll
	clang main.s -o main

// this file is used for testing the LLVM IR

extern "C" // disables name mangling
    int
    add(int a, int b) {
  return a + b;
}

extern "C" int sub(int a, int b) { return a - b; }

extern "C" int mul(int a, int b) { return a * b; }

extern "C" int div(int a, int b) { return a / b; }

extern "C" int pow(int base, int exp) {
  if (exp == 1) {
    return base;
  } else {
    return pow(base * base, exp - 1);
  }
}

int main() { return pow(3, 2); }

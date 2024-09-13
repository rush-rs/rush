extern void exit(int);
extern int __rush_internal_pow_int(int, int);

extern char __rush_internal_cast_int_to_char(int);
extern char __rush_internal_cast_float_to_char(double);

void _start() {
  int pow_res = __rush_internal_pow_int(2, 7);
  char char_1 = __rush_internal_cast_int_to_char(197);
  char char_2 = __rush_internal_cast_float_to_char(197.0);
  exit(char_2);
}

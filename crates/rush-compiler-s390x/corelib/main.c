extern void exit(int);
extern long __rush_internal_pow_int(long, long);

extern char __rush_internal_cast_int_to_char(long);
extern char __rush_internal_cast_float_to_char(double);

void _start() {
    int pow_res = __rush_internal_pow_int(2, 7);
    char char_1 = __rush_internal_cast_int_to_char(197);
    char char_2 = __rush_internal_cast_float_to_char(197.0);
    exit(pow_res);
}

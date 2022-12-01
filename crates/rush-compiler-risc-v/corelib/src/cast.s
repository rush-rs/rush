# This file is part of the rush corelib for RISC-V
# - Authors: MikMuellerDev
# This file, alongside others is linked to form a RISC-V targeted rush program.

# Converts a i64 into a u8 char.
# If the source is < 0, the result will be 0.
# Furthermore, if the source is > 127, the result will be 127.
# fn __rush_internal_cast_int_to_char(from: int) -> char
.global __rush_internal_cast_int_to_char

__rush_internal_cast_int_to_char:
	# if from < 0; return 0
	blt a0, zero, return_0
	# if from > 127; return 127
	li t0, 127
	bgt a0, t0, return_127
	ret	# return argument


# Converts a f64 into a u8 char.
# If the source is < 0.0, the result will be 0.
# Furthermore, if the source is > 127.0, the result will be 127.
# fn __rush_internal_cast_float_to_char(from: float) -> char
.global __rush_internal_cast_float_to_char

__rush_internal_cast_float_to_char:
	# if from < 0.0; return 0
	fld ft0, float_0, t0
	flt.d t0, fa0, ft0
	bne t0, zero, return_0
	# if from > 127.0; return 127
	fld ft0, float_127, t0
	fgt.d t0, fa0, ft0
	bne t0, zero, return_127
	# truncate argument to int
	fcvt.w.d a0, fa0, rdn
	ret # return truncated value


### UTILS ###
return_0:
	li a0, 0
	ret

return_127:
	li a0, 127
	ret

### CONSTANTS ###
.section .rodata
float_0:
	.dword 0x0000000000000000

float_127:
	.dword 0x405fc00000000000

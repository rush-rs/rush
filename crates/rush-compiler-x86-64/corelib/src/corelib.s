.intel_syntax

.global __rush_internal_cast_int_to_char
.global __rush_internal_cast_float_to_char
.global __rush_internal_pow_int
.global exit

###############################
########## CONSTANTS ##########
###############################

.section .rodata

float_0:
	.quad 0x0000000000000000

float_127:
	.quad 0x405fc00000000000

.section .text

###########################
########## UTILS ##########
###########################

return_0:
	mov %rax, 0
	ret

return_127:
	mov %rax, 127
	ret

###############################
########## FUNCTIONS ##########
###############################

# fn __rush_internal_cast_int_to_char(input: int) -> char {
__rush_internal_cast_int_to_char:
	# if input < 0 { return '\x00'; }
	cmp %rdi, 0
	jl return_0

	# if input > 0x7F { return '\x7F'; }
	cmp %rdi, 0x7F
	jg return_127

	# return input;
	mov %rax, %rdi
	ret
# }

# fn __rush_internal_cast_float_to_char(input: float) -> char {
__rush_internal_cast_float_to_char:
	# if input < 0 { return '\x00'; }
	ucomisd %xmm0, qword ptr [%rip + float_0]
	jb return_0

	# if input > 0x7F { return '\x7F'; }
	ucomisd %xmm0, qword ptr [%rip + float_127]
	ja return_127

	# return input as int;
	cvttsd2si %rax, %xmm0
	ret
# }

# fn __rush_internal_pow_int(mut base: int, mut exponent: int) -> int {
__rush_internal_pow_int:
	# if exponent < 0 { return 0; }
	cmp %rsi, 0
	jl return_0

	# let mut accumulator = 1;
	mov %rax, 1

	# loop {
	loop:
		# if exponent <= 1 { break; }
		cmp %rsi, 1
		jle after_loop

		# if (exponent & 1) == 1 { accumulator *= base; }
		mov %rdx, %rsi
		and %rdx, 1
		cmp %rdx, 1
		jne after_if
		## accumulator *= base
		imul %rax, %rdi
	after_if:

		# exponent >> 1;
		shr %rsi, 1

		# base *= base;
		imul %rdi, %rdi

		# continue;
		jmp loop
	after_loop:
	# }

	# accumulator *= base;
	imul %rax, %rdi

	# return accumulator;
	ret
# }

# fn exit(code: int) {
exit:
	mov %rax, 60	# 60 = sys_exit
	syscall			# exit code is already set in %rdi
# }

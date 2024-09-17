	.file	"main.c"
	.machinemode zarch
	.machine "z900"
.text
	.align	8
.globl a
	.type	a, @function
a:
.LFB0:
	.cfi_startproc
	stmg	%r11,%r15,88(%r15)
	.cfi_offset 11, -72
	.cfi_offset 12, -64
	.cfi_offset 13, -56
	.cfi_offset 14, -48
	.cfi_offset 15, -40
	larl	%r13,.L5
	aghi	%r15,-192
	.cfi_def_cfa_offset 352
	lgr	%r11,%r15
	.cfi_def_cfa_register 11
	ld	%f0,.L6-.L5(%r13)
	std	%f0,168(%r11)
	lhi	%r1,2
	st	%r1,164(%r11)
	j	.L2
.L3:
	lgf	%r1,164(%r11)
	lghi	%r3,3
	lgr	%r2,%r1
	brasl	%r14,__rush_internal_pow_int@PLT
	lgr	%r1,%r2
	cdgbr	%f2,%r1
	ld	%f0,.L6-.L5(%r13)
	ddbr	%f0,%f2
	adb	%f0,168(%r11)
	std	%f0,168(%r11)
	l	%r1,164(%r11)
	ahi	%r1,1
	st	%r1,164(%r11)
.L2:
	l	%r1,164(%r11)
	chi	%r1,29999
	jle	.L3
	ld	%f0,.L7-.L5(%r13)
	std	%f0,176(%r11)
	ld	%f0,168(%r11)
	mdb	%f0,176(%r11)
	cgdbr	%r1,5,%f0
	stg	%r1,184(%r11)
	lg	%r1,184(%r11)
	lg	%r2,.L8-.L5(%r13)
	lgr	%r5,%r2
	mlgr	%r4,%r1
	lgr	%r2,%r4
	lgr	%r3,%r5
	srag	%r4,%r1,63
	ng	%r4,.L8-.L5(%r13)
	sgr	%r2,%r4
	lg	%r4,.L8-.L5(%r13)
	srag	%r4,%r4,63
	ngr	%r4,%r1
	sgr	%r2,%r4
	srag	%r2,%r2,18
	srag	%r3,%r1,63
	lgr	%r1,%r2
	sgr	%r1,%r3
	lgr	%r2,%r1
	lg	%r4,304(%r11)
	lmg	%r11,%r15,280(%r11)
	.cfi_restore 15
	.cfi_restore 14
	.cfi_restore 13
	.cfi_restore 12
	.cfi_restore 11
	.cfi_def_cfa 15, 160
	br	%r4
	.section	.rodata
	.align	8
.L5:
.L7:
	.long	1097011920
	.long	0
.L6:
	.long	1072693248
	.long	0
.L8:
	.quad	4835703278458516699
	.align	2
.text
	.cfi_endproc
.LFE0:
	.size	a, .-a
	.align	8
.globl _start
	.type	_start, @function
_start:
.LFB1:
	.cfi_startproc
	stmg	%r11,%r15,88(%r15)
	.cfi_offset 11, -72
	.cfi_offset 12, -64
	.cfi_offset 13, -56
	.cfi_offset 14, -48
	.cfi_offset 15, -40
	aghi	%r15,-160
	.cfi_def_cfa_offset 320
	lgr	%r11,%r15
	.cfi_def_cfa_register 11
	brasl	%r14,a@PLT
	lgr	%r1,%r2
	lgfr	%r1,%r1
	lgr	%r2,%r1
	brasl	%r14,exit@PLT
	.cfi_endproc
.LFE1:
	.size	_start, .-_start
	.ident	"GCC: (GNU) 13.2.0"
	.section	.note.GNU-stack,"",@progbits

.section .text
.global exit

.type exit, @function
exit:
    movl $1, %eax
    movl %edi, %ebx
    int $0x80

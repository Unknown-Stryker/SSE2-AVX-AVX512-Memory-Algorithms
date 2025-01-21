# Copyright © from 2023 to current, UNKNOWN STRYKER. All Rights Reserved.
# This code is a part of Project Frogman, and the part of Frogman Engine memcpy implementation.

# References used to implement void __x86_64_AVX_SSE_unaligned_memcpy(void* dest_p, const void* source_p, size_t bytes_to_copy_p):
# https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX_ALL&text=_mm256_store_si256&ig_expand=6491,6548,6491
# https://staffwww.fullcoll.edu/aclifton/cs241/lecture-asm-to-c-interop.html
# http://6.s081.scripts.mit.edu/sp18/x86-64-architecture-guide.html
# https://aaronbloomfield.github.io/pdr/book/x86-64bit-ccc-chapter.pdf
# https://ftp.gnu.org/old-gnu/Manuals/gas-2.9.1/html_chapter/as_toc.html
# https://en.wikipedia.org/wiki/X86_calling_conventions#Register_preservation
# https://godbolt.org/
# https://github.com/Frogman-Engine/Frogman-Engine/blob/chaos/SDK/Core/Include/FE/memory.hxx

# x86-64 AT&T assembly register look-up table.
# https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf 
# qword  dword  word   byte
# %rax   %eax   %ax    %al
# %rcx   %ecx   %cx    %cl
# %rdx   %edx   %dx    %dl
# %rbx   %ebx   %bx    %bl
# %rsi   %esi   %si    %sil
# %rdi   %edi   %di    %dil
# %rsp   %esp   %sp    %spl
# %rbp   %ebp   %bp    %bpl
# %r8    %r8d   %r8w   %r8b
# %r9    %r9d   %r9w   %r9b
# %r10   %r10d  %r10w  %r10b
# %r11   %r11d  %r11w  %r11b
# %r12   %r12d  %r12w  %r12b
# %r13   %r13d  %r13w  %r13b
# %r14   %r14d  %r14w  %r14b
# %r15   %r15d  %r15w  %r15b

# X86-64 calling convention parameter order:
# return: %rax
# first: %rdi
# second: %rsi
# third: %rdx
# fourth: %rcx
# fifth: %r8
# sixth: %r9




.section .data # A non-global data section.
# A jump table for qword-sized code pointer switch case labels.
# switch_case is the name of qword pointer array, which means the address of switch_case is equal to the the address of case0! Don't lie, Copliot. Haha!
switch_case: 
    .quad case0
    .quad case1
    .quad case2
    .quad case3
    .quad case4
    .quad case5
    .quad case6
    .quad case7
    .quad case8
    .quad case9
    .quad case10
    .quad case11
    .quad case12
    .quad case13
    .quad case14
    .quad case15
    .quad case16
    .quad case17
    .quad case18
    .quad case19
    .quad case20
    .quad case21
    .quad case22
    .quad case23
    .quad case24
    .quad case25
    .quad case26
    .quad case27
    .quad case28
    .quad case29
    .quad case30
    .quad case31


# void* dest_p: %rdi
# const void* source_p: %rsi
# size_t bytes_to_copy_p: %rdx 
.global __x86_64_AVX_SSE_unaligned_memcpy # Declares the name as a global symbol, making it accessible to other parts of the program.


.section .text # Marks the beginning of the code section where executable instructions are defined.
__x86_64_AVX_SSE_unaligned_memcpy:
    # %rip is the instruction pointer.
    # The code below is '%rcx = &switch_case;'
    leaq switch_case(%rip), %rcx # Load the base address of the switch case label onto %rcx, which is the fourth parameter according to the x86-64 C calling convention.
    # The fourth argument space is unoccupied. Therefore, it can store the address value in it.
    
    # So the %r8, the fifth argument.
    movq %rdx, %r8 # Store the value of bytes_to_copy_p to %r8 to calculate the required AVX operations count.
    sarq $5, %r8 # Since 2^5 is 32, '%r8 >>= 5' results the same as '%r8 /= 32.' Thus will give %r8 the required AVX operations count.

    movq %r8, %r9 # Preserve it in %r9.

    cmpq $0, %r8 # Compare the qword integer 0 with %r8. 
    jle copy_leftover_bytes # jump if (0 >= %r8)

avx_for_loop: # for (; 0 < %r8; --%r8)
    cmpq $0, %r8 # Compare the qword integer 0 with %r8.

    vmovdqu (%rsi), %ymm0 # Dereference %rsi and load the data into AVX 256-bit %ymm vector register.
    vmovdqu %ymm0, (%rdi) # Store the loaded 256 bits data into the dereferenced %rdi.
    addq $32, %rdi # Increment the void* by 32 bytes. %rdi += 32;
    addq $32, %rsi # Increment the void* by 32 byteses. %rsi += 32;

    subq $1, %r8 # %r8 -= 1;
    # 0 < %r8. Jump if %r8 is greater than 0. 
    ja avx_for_loop # Note: jge is for signed, and jae is for unsigned.
 
    imulq $32, %r9 # Get the byte size of the memory being copied.
    subq %r9, %rdx # Subtract bytes_to_copy_p: %rdx by %r9.

copy_leftover_bytes:
    # %rcx: the base label address, %rdx: the size_t argument, %r8: the register to store the address of switch_case[index].
    jmp *(%rcx, %rdx, 8) # Jump to the address %rcx + (%rdx * 8).

# jmp x 4
case31: # rax, eax, ax registers are used to store a function return value in C language.
    # It is fine to use the register for the memcpy operation because the function does not return any.
    movq (%rsi), %rax # Dereference %rsi and load the data into qword %rax register.
    movq %rax, (%rdi) # Store the loaded 64 bits data into the dereferenced %rdi.
    addq $8, %rdi # Increment the void* by 8 bytes.
    addq $8, %rsi # Increment the void* by 8 bytes.
    jmp case23

# jmp x 3
case30:
    movw (%rsi), %ax # Dereference %rsi and load the data into word %ax register.
    movw %ax, (%rdi) # Store the loaded 16 bits data into the dereferenced %rdi.
    addq $2, %rdi # Increment the void* by 2 bytes.
    addq $2, %rsi # Increment the void* by 2 bytes.
    jmp case28

# jmp x 3
case29:
    movb (%rsi), %al # Dereference %rsi and load the data into byte %al register.
    movb %al, (%rdi) # Store the loaded 8 bits data into the dereferenced %rdi.
    addq $1, %rdi # Increment the void* by 1 bytes.
    addq $1, %rsi # Increment the void* by 1 bytes.
    jmp case28

# jmp x 2
case28:
    movdqu (%rsi), %xmm0 # Dereference %rsi and load the data into SSE2 128-bit %xmm vector register.
    movdqu %xmm0, (%rdi) # Store the loaded 128 bits data into the dereferenced %rdi.
    addq $16, %rdi # Increment the void* by 16 bytes.
    addq $16, %rsi # Increment the void* by 16 bytes.
    jmp case12

# jmp x 3
case27:
    movb (%rsi), %al # Dereference %rsi and load the data into byte %al register.
    movb %al, (%rdi) # Store the loaded 8 bits data into the dereferenced %rdi.
    addq $1, %rdi # Increment the void* by 1 bytes.
    addq $1, %rsi # Increment the void* by 1 bytes.
    jmp case26

# jmp x 2
case26:
    movw (%rsi), %ax # Dereference %rsi and load the data into word %ax register.
    movw %ax, (%rdi) # Store the loaded 16 bits data into the dereferenced %rdi.
    addq $2, %rdi # Increment the void* by 2 bytes.
    addq $2, %rsi # Increment the void* by 2 bytes.
    jmp case24

# jmp x 2
case25:
    movb (%rsi), %al # Dereference %rsi and load the data into byte %al register.
    movb %al, (%rdi) # Store the loaded 8 bits data into the dereferenced %rdi.
    addq $1, %rdi # Increment the void* by 1 bytes.
    addq $1, %rsi # Increment the void* by 1 bytes.
    jmp case24

# jmp x 2
case24:
    movdqu (%rsi), %xmm0 # Dereference %rsi and load the data into SSE2 128-bit %xmm vector register.
    movdqu %xmm0, (%rdi) # Store the loaded 128 bits data into the dereferenced %rdi.
    addq $16, %rdi # Increment the void* by 16 bytes.
    addq $16, %rsi # Increment the void* by 16 bytes.
    jmp case8 

# jmp x 3
case23:
    movdqu (%rsi), %xmm0 # Dereference %rsi and load the data into SSE2 128-bit %xmm vector register.
    movdqu %xmm0, (%rdi) # Store the loaded 128 bits data into the dereferenced %rdi.
    addq $16, %rdi # Increment the void* by 16 bytes.
    addq $16, %rsi # Increment the void* by 16 bytes.
    jmp case7

# jmp x 2
case22:
    movdqu (%rsi), %xmm0 # Dereference %rsi and load the data into SSE2 128-bit %xmm vector register.
    movdqu %xmm0, (%rdi) # Store the loaded 128 bits data into the dereferenced %rdi.
    addq $16, %rdi # Increment the void* by 16 bytes.
    addq $16, %rsi # Increment the void* by 16 bytes.
    jmp case6

# jmp x 2
case21:
    movdqu (%rsi), %xmm0 # Dereference %rsi and load the data into SSE2 128-bit %xmm vector register.
    movdqu %xmm0, (%rdi) # Store the loaded 128 bits data into the dereferenced %rdi.
    addq $16, %rdi # Increment the void* by 16 bytes.
    addq $16, %rsi # Increment the void* by 16 bytes.
    jmp case5

# jmp x 2
case20:
    movl (%rsi), %eax # Dereference %rsi and load the data into dword %eax register.
    movl %eax, (%rdi) # Store the loaded 32 bits data into the dereferenced %rdi.
    addq $4, %rdi # Increment the void* by 4 bytes.
    addq $4, %rsi # Increment the void* by 4 bytes.
    jmp case16

# jmp x 2
case19:
    movdqu (%rsi), %xmm0 # Dereference %rsi and load the data into SSE2 128-bit %xmm vector register.
    movdqu %xmm0, (%rdi) # Store the loaded 128 bits data into the dereferenced %rdi.
    addq $16, %rdi # Increment the void* by 16 bytes.
    addq $16, %rsi # Increment the void* by 16 bytes.
    jmp case3

# jmp x 1
case18:
    movw (%rsi), %ax # Dereference %rsi and load the data into word %ax register.
    movw %ax, (%rdi) # Store the loaded 16 bits data into the dereferenced %rdi.
    addq $2, %rdi # Increment the void* by 2 bytes.
    addq $2, %rsi # Increment the void* by 2 bytes.
    jmp case16

# jmp x 1
case17:
    movb (%rsi), %al # Dereference %rsi and load the data into byte %al register.
    movb %al, (%rdi) # Store the loaded 8 bits data into the dereferenced %rdi.
    addq $1, %rdi # Increment the void* by 1 bytes.
    addq $1, %rsi # Increment the void* by 1 bytes.
    jmp case16

# jmp x 0
case16:
    movdqu (%rsi), %xmm0 # Dereference %rsi and load the data into SSE2 128-bit %xmm vector register.
    movdqu %xmm0, (%rdi) # Store the loaded 128 bits data into the dereferenced %rdi.
    ret

# jmp x 3
case15:
    movq (%rsi), %rax # Dereference %rsi and load the data into qword %rax register.
    movq %rax, (%rdi) # Store the loaded 64 bits data into the dereferenced %rdi.
    addq $8, %rdi # Increment the void* by 8 bytes.
    addq $8, %rsi # Increment the void* by 8 bytes.
    jmp case7

# jmp x 2
case14:
    movq (%rsi), %rax # Dereference %rsi and load the data into qword %rax register.
    movq %rax, (%rdi) # Store the loaded 64 bits data into the dereferenced %rdi.
    addq $8, %rdi # Increment the void* by 8 bytes.
    addq $8, %rsi # Increment the void* by 8 bytes.
    jmp case6

# jmp x 2
case13:
    movq (%rsi), %rax # Dereference %rsi and load the data into qword %rax register.
    movq %rax, (%rdi) # Store the loaded 64 bits data into the dereferenced %rdi.
    addq $8, %rdi # Increment the void* by 8 bytes.
    addq $8, %rsi # Increment the void* by 8 bytes.
    jmp case5

# jmp x 1
case12:
    movl (%rsi), %eax # Dereference %rsi and load the data into dword %eax register.
    movl %eax, (%rdi) # Store the loaded 32 bits data into the dereferenced %rdi.
    addq $4, %rdi # Increment the void* by 4 bytes.
    addq $4, %rsi # Increment the void* by 4 bytes.
    jmp case8

# jmp x 2
case11:
    movq (%rsi), %rax # Dereference %rsi and load the data into qword %rax register.
    movq %rax, (%rdi) # Store the loaded 64 bits data into the dereferenced %rdi.
    addq $8, %rdi # Increment the void* by 8 bytes.
    addq $8, %rsi # Increment the void* by 8 bytes.
    jmp case3

# jmp x 1
case10:
    movw (%rsi), %ax # Dereference %rsi and load the data into word %ax register.
    movw %ax, (%rdi) # Store the loaded 16 bits data into the dereferenced %rdi.
    addq $2, %rdi # Increment the void* by 2 bytes.
    addq $2, %rsi # Increment the void* by 2 bytes.
    jmp case8

# jmp x 1
case9:
    movb (%rsi), %al # Dereference %rsi and load the data into byte %al register.
    movb %al, (%rdi) # Store the loaded 8 bits data into the dereferenced %rdi.
    addq $1, %rdi # Increment the void* by 1 bytes.
    addq $1, %rsi # Increment the void* by 1 bytes.
    jmp case8

# jmp x 0
case8:
    movq (%rsi), %rax # Dereference %rsi and load the data into qword %rax register.
    movq %rax, (%rdi) # Store the loaded 64 bits data into the dereferenced %rdi.
    ret

# jmp x 2
case7:
    movb (%rsi), %al # Dereference %rsi and load the data into byte %al register.
    movb %al, (%rdi) # Store the loaded 8 bits data into the dereferenced %rdi.
    addq $1, %rdi # Increment the void* by 1 bytes.
    addq $1, %rsi # Increment the void* by 1 bytes.
    jmp case6

# jmp x 1
case6:
    movw (%rsi), %ax # Dereference %rsi and load the data into word %ax register.
    movw %ax, (%rdi) # Store the loaded 16 bits data into the dereferenced %rdi.
    addq $2, %rdi # Increment the void* by 2 bytes.
    addq $2, %rsi # Increment the void* by 2 bytes.
    jmp case4

# jmp x 1
case5:
    movb (%rsi), %al # Dereference %rsi and load the data into byte %al register.
    movb %al, (%rdi) # Store the loaded 8 bits data into the dereferenced %rdi.
    addq $1, %rdi # Increment the void* by 1 bytes.
    addq $1, %rsi # Increment the void* by 1 bytes.
    jmp case4

# jmp x 0
case4:
    movl (%rsi), %eax # Dereference %rsi and load the data into dword %eax register.
    movl %eax, (%rdi) # Store the loaded 32 bits data into the dereferenced %rdi.
    ret

# jmp x 1
case3:
    movb (%rsi), %al # Dereference %rsi and load the data into byte %al register.
    movb %al, (%rdi) # Store the loaded 8 bits data into the dereferenced %rdi.
    addq $1, %rdi # Increment the void* by 1 bytes.
    addq $1, %rsi # Increment the void* by 1 bytes.
    jmp case2

# jmp x 0
case2:
    movw (%rsi), %ax # Dereference %rsi and load the data into word %ax register.
    movw %ax, (%rdi) # Store the loaded 16 bits data into the dereferenced %rdi.
    ret

# jmp x 0
case1:
    movb (%rsi), %al # Dereference %rsi and load the data into byte %al register.
    movb %al, (%rdi) # Store the loaded 8 bits data into the dereferenced %rdi.
    ret

# jmp x 0
case0:
    ret

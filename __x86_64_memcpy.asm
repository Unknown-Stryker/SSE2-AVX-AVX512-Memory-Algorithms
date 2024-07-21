# Copyright © from 2023 to current, UNKNOWN STRYKER. All Rights Reserved.
# This code is a part of Project Frogman, and the part of Frogman Engine memcpy implementation.

# References used to implement void __x86_64_memcpy(void* dest_p, const void* source_p, size_t bytes_to_copy_p):
# https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX_ALL&text=_mm256_store_si256&ig_expand=6491,6548,6491
# https://staffwww.fullcoll.edu/aclifton/cs241/lecture-asm-to-c-interop.html
# http://6.s081.scripts.mit.edu/sp18/x86-64-architecture-guide.html
# https://aaronbloomfield.github.io/pdr/book/x86-64bit-ccc-chapter.pdf
# https://ftp.gnu.org/old-gnu/Manuals/gas-2.9.1/html_chapter/as_toc.html

# x86-64 AT&T assembly register look-up table
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
# first: %rdi
# second: %rsi
# third: %rdx
# fourth: %rcx
# fifth: %r8
# sixth: %r9

# void* dest_p: %rdi
# const void* source_p: %rsi
# size_t bytes_to_copy_p: %rdx 
.global __x86_64_memcpy # Declares the name as a global symbol, making it accessible to other parts of the program.

.data # a jump table for qword-sized code pointer switch case labels
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

.text # Marks the beginning of the code section where executable instructions are defined.
__x86_64_memcpy:
    # %rip is the instruction pointer.
    leaq switch_case(%rip), %rcx # load the base address of the switch case label onto %rcx, which is the fourth parameter according to the x86-64 C calling convention.
    # The fourth argument space is unused by this function parameters. Therefore, it can store the address value in it.
    
    cmpq $32, %rdx # Compare the qword integer 32 with %rdx.
    jae default_case # copy 256 bits using avx if %rdx is greater or equal to 32. Note: jge is for signed, and jae is for unsigned.
 
 __x86_64_memcpy_loop:
    # %rcx: the base label address, %rdx: the size_t argument, %r8: the register to store the address of switch_case[index].
    jmp *(%rcx, %rdx, 8) # Jump to the address %rcx + (%rdx * 8).


default_case:
    vmovdqu (%rsi), %ymm0 # Dereference %rsi and load the data into AVX 256-bit %ymm vector register.
    vmovdqu %ymm0, (%rdi) # Store the loaded 256 bits data into the dereferenced %rdi.
    addq $32, %rdi # Increment the void* by 32 bytes
    addq $32, %rsi # Increment the void* by 32 bytes
    subq $32, %rdx # Decrement the leftover size to copy by 32 bytes
    cmpq $32, %rdx # Compare the qword integer 32 with %rdx.
    jae default_case # copy 256 bits using avx if %rdx is greater or equal to 32.
    jmp __x86_64_memcpy_loop
case31:
case30:
case29:
case28:
case27:
case26:
case25:
case24:
case23:
case22:
case21:
case20:
case19:
case18:
case17:
case16:
    movdqu (%rsi), %xmm0 # Dereference %rsi and load the data into SSE2 128-bit %xmm vector register.
    movdqu %xmm0, (%rdi) # Store the loaded 128 bits data into the dereferenced %rdi.
    addq $16, %rdi # Increment the void* by 16 bytes
    addq $16, %rsi # Increment the void* by 16 bytes
    subq $16, %rdx # Decrement the size to copy by 16 bytes
    jmp __x86_64_memcpy_loop
case15:
case14:
case13:
case12:
case11:
case10:
case9:
case8:
    movq (%rsi), %rax # Dereference %rsi and load the data into qword %rax register.
    movq %rax, (%rdi) # Store the loaded 64 bits data into the dereferenced %rdi.
    addq $8, %rdi # Increment the void* by 8 bytes
    addq $8, %rsi # Increment the void* by 8 bytes
    subq $8, %rdx # Decrement the size to copy by 8 bytes
    jmp __x86_64_memcpy_loop
case7:
case6:
case5:
case4:
    movl (%rsi), %eax # Dereference %rsi and load the data into dword %eax register.
    movl %eax, (%rdi) # Store the loaded 32 bits data into the dereferenced %rdi.
    addq $4, %rdi # Increment the void* by 4 bytes
    addq $4, %rsi # Increment the void* by 4 bytes
    subq $4, %rdx # Decrement the size to copy by 4 bytes
    jmp __x86_64_memcpy_loop
case3:
case2:
    movw (%rsi), %ax # Dereference %rsi and load the data into word %ax register.
    movw %ax, (%rdi) # Store the loaded 16 bits data into the dereferenced %rdi.
    addq $2, %rdi # Increment the void* by 2 bytes
    addq $2, %rsi # Increment the void* by 2 bytes
    subq $2, %rdx # Decrement the size to copy by 2 bytes
    jmp __x86_64_memcpy_loop
case1:
    movb (%rsi), %al # Dereference %rsi and load the data into byte %al register.
    movb %al, (%rdi) # Store the loaded 8 bits data into the dereferenced %rdi.
case0:
    ret
    

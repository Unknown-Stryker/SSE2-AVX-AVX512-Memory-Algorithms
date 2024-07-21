#ifndef __X86_64_MEMCPY_H_
#define __X86_64_MEMCPY_H_
#include <cstddef>
/*
Copyright © from 2023 to current, UNKNOWN STRYKER. All Rights Reserved.

It was written in AT&T assembly to outperform and replace the C++ implementation of FE::memcpy.
Sadly, it could not run fater than the optimized C++ code with clang++-12 -O3.
It was worth spending time and efforts to learn how C functions work on x86-64 CPU architecture.

Those three in-house memory algorithms have reached their performance goal by removing local variables.
It was quiet simpiler to achieve than I thought it would be, and I admit that clang outsmarts me. Ha ha ha.

- Release Build -
2024-06-30T09:14:35+00:00
Running /workspace/Frogman-Engine-Lab/Frogman-Engine-Tests/FE-Tests/Unit-Tests/FE.core.memory/FE.core.memory_test
Run on (16 X 2799.99 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 512 KiB (x8)
  L3 Unified 16384 KiB (x2)
Load Average: 3.95, 2.41, 1.56
-----------------------------------------------------------------------------------------
Benchmark                                               Time             CPU   Iterations
-----------------------------------------------------------------------------------------
x86_64_aligned_memcpy_written_in_asm_benchmark       3911 ns         3911 ns       176432
FE_aligned_memcpy_benchmark                          2063 ns         2063 ns       333200
std_memcpy_benchmark                                 2076 ns         2076 ns       340406
FE_aligned_memmove_benchmark                         98.3 ns         98.3 ns      7187033
std_memmove_benchmark                                99.8 ns         99.8 ns      6786284
FE_aligned_memset_benchmark                          1030 ns         1030 ns       683220
std_memset_benchmark                                 1023 ns         1023 ns       675393
*/

extern "C" void __x86_64_memcpy(void* dest_p, const void* source_p, size_t bytes_to_copy_p);

/*
- Debug Build -
2024-06-30T10:34:43+00:00
Running /workspace/Frogman-Engine-Lab/Frogman-Engine-Tests/FE-Tests/Unit-Tests/FE.core.memory/FE.core.memory_test
Run on (16 X 2799.99 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 512 KiB (x8)
  L3 Unified 16384 KiB (x2)
Load Average: 1.92, 1.72, 1.44
***WARNING*** Library was built as DEBUG. Timings may be affected.
-----------------------------------------------------------------------------------------
Benchmark                                               Time             CPU   Iterations
-----------------------------------------------------------------------------------------
x86_64_aligned_memcpy_written_in_asm_benchmark       3926 ns         3927 ns       176643
FE_aligned_memcpy_benchmark                         18938 ns        18939 ns        36683
std_memcpy_benchmark                                 3601 ns         3601 ns       193500
FE_aligned_memmove_benchmark                          999 ns          999 ns       702971
std_memmove_benchmark                                 103 ns          103 ns      6832297
FE_aligned_memset_benchmark                        126545 ns       126546 ns         5563
std_memset_benchmark                                 2060 ns         2060 ns       341069
*/
#endif

# Performance Benchmark 
Release Build with LLVM Clang 12 C++
````
Running a test named: /workspace/Frogman-Engine-Lab/Frogman-Engine-Tests/FE-Tests/Unit-Tests/FE.core.memory/FE.core.memory_test
[==========] Running 4 tests from 3 test suites.
[----------] Global test environment set-up.
[----------] 2 tests from memmove
[ RUN      ] memmove.string_insertion
[       OK ] memmove.string_insertion (0 ms)
[ RUN      ] memmove.General
[       OK ] memmove.General (0 ms)
[----------] 2 tests from memmove (0 ms total)

[----------] 1 test from memcpy
[ RUN      ] memcpy._
[       OK ] memcpy._ (0 ms)
[----------] 1 test from memcpy (0 ms total)

[----------] 1 test from memset
[ RUN      ] memset._
[       OK ] memset._ (0 ms)
[----------] 1 test from memset (0 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 3 test suites ran. (0 ms total)
[  PASSED  ] 4 tests.


2024-07-21T05:28:50+00:00
Running /workspace/Frogman-Engine-Lab/Frogman-Engine-Tests/FE-Tests/Unit-Tests/FE.core.memory/FE.core.memory_test
Run on (16 X 2799.99 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 512 KiB (x8)
  L3 Unified 16384 KiB (x2)
Load Average: 4.73, 4.87, 3.43
-----------------------------------------------------------------------
Benchmark                             Time             CPU   Iterations
-----------------------------------------------------------------------
FE_aligned_memcpy_benchmark        2058 ns         2058 ns       340778
std_memcpy_benchmark               2057 ns         2057 ns       337832
FE_aligned_memmove_benchmark       97.6 ns         97.6 ns      7139920
std_memmove_benchmark              99.4 ns         99.4 ns      7065484
FE_aligned_memset_benchmark        1030 ns         1030 ns       682759
std_memset_benchmark               1032 ns         1032 ns       681416
````

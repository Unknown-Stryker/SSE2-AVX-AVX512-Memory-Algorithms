/*
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh <version>
clang-<version> --version
*/
#include "memcpy.h"
#include <benchmark/benchmark.h>
#include <cstring>

// Size of the buffer to copy
constexpr size_t BUFFER_SIZE = 1024 * 1024; // 1 MB

// Buffers
alignas(32) char src[BUFFER_SIZE];
alignas(32) char dst[BUFFER_SIZE];




static void BM_memcpy(benchmark::State& state) 
{
    for (auto _ : state) 
    {
        std::memcpy(dst, src, BUFFER_SIZE);
        benchmark::ClobberMemory(); // Ensure the compiler doesn't optimize away the memcpy
    }
}
BENCHMARK(BM_memcpy);


static void BM__x86_64_AVX_SSE_aligned_memcpy(benchmark::State& state) 
{
    for (auto _ : state) 
    {
        __x86_64_AVX_SSE_aligned_memcpy(dst, src, BUFFER_SIZE);
        benchmark::ClobberMemory(); // Ensure the compiler doesn't optimize away the memcpy
    }
}
BENCHMARK(BM__x86_64_AVX_SSE_aligned_memcpy);

static void BM__x86_64_AVX_SSE_unaligned_memcpy(benchmark::State& state) 
{
    for (auto _ : state) 
    {
        __x86_64_AVX_SSE_unaligned_memcpy(dst, src, BUFFER_SIZE);
        benchmark::ClobberMemory(); // Ensure the compiler doesn't optimize away the memcpy
    }
}
BENCHMARK(BM__x86_64_AVX_SSE_unaligned_memcpy);

static void BM__x86_64_AVX_SSE_dest_aligned_memcpy(benchmark::State& state) 
{
    for (auto _ : state) 
    {
        __x86_64_AVX_SSE_dest_aligned_memcpy(dst, src, BUFFER_SIZE);
        benchmark::ClobberMemory(); // Ensure the compiler doesn't optimize away the memcpy
    }
}
BENCHMARK(BM__x86_64_AVX_SSE_dest_aligned_memcpy);

static void BM__x86_64_AVX_SSE_source_aligned_memcpy(benchmark::State& state) 
{
    for (auto _ : state) 
    {
        __x86_64_AVX_SSE_source_aligned_memcpy(dst, src, BUFFER_SIZE);
        benchmark::ClobberMemory(); // Ensure the compiler doesn't optimize away the memcpy
    }
}
BENCHMARK(BM__x86_64_AVX_SSE_source_aligned_memcpy);




static void x86_64_AVX_SSE_memcpy_test()
{
    for (int i = 32; 0 < i; --i)
    {
        //__x86_64_AVX_SSE_memcpy(dst, src, i);
        //benchmark::ClobberMemory(); // Ensure the compiler doesn't optimize away the memcpy
    }
}




int main(int argc, char** argv) 
{          
    //x86_64_AVX_SSE_memcpy_test();  

    char arg0_default[] = "benchmark";                                  
    char* args_default = arg0_default;         

    if (!argv) 
    {                                                        
      argc = 1;                                                         
      argv = &args_default;                                             
    }                                                                   
    ::benchmark::Initialize(&argc, argv);  

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) 
        return 1; 
    
    ::benchmark::RunSpecifiedBenchmarks();                              
    ::benchmark::Shutdown();                                            
    return 0;                                                           
}   

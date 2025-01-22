/*
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh <version>
clang-<version> --version
*/
#include "memcpy.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include <cstring>


// Size of the buffer to copy
constexpr size_t BUFFER_SIZE = 1024 * 1024; // 1 MB




static void BM_memcpy(benchmark::State& state) 
{
    // Buffers
    alignas(32) char src[BUFFER_SIZE];
    alignas(32) char dst[BUFFER_SIZE];

    for (auto _ : state) 
    {
        std::memcpy(dst, src, BUFFER_SIZE);
        benchmark::ClobberMemory(); // Ensure the compiler doesn't optimize away the memcpy
    }
}
BENCHMARK(BM_memcpy);


static void BM__x86_64_AVX_SSE_aligned_memcpy(benchmark::State& state) 
{
    // Buffers
    alignas(32) char src[BUFFER_SIZE];
    alignas(32) char dst[BUFFER_SIZE];

    for (auto _ : state) 
    {
        __x86_64_AVX_SSE_aligned_memcpy(dst, src, BUFFER_SIZE);
        benchmark::ClobberMemory(); // Ensure the compiler doesn't optimize away the memcpy
    }
}
BENCHMARK(BM__x86_64_AVX_SSE_aligned_memcpy);

static void BM__x86_64_AVX_SSE_unaligned_memcpy(benchmark::State& state) 
{
    // Buffers
    alignas(32) char src[BUFFER_SIZE];
    alignas(32) char dst[BUFFER_SIZE];

    for (auto _ : state) 
    {
        __x86_64_AVX_SSE_unaligned_memcpy(dst, src, BUFFER_SIZE);
        benchmark::ClobberMemory(); // Ensure the compiler doesn't optimize away the memcpy
    }
}
BENCHMARK(BM__x86_64_AVX_SSE_unaligned_memcpy);

static void BM__x86_64_AVX_SSE_dest_aligned_memcpy(benchmark::State& state) 
{
    // Buffers
    alignas(32) char src[BUFFER_SIZE];
    alignas(32) char dst[BUFFER_SIZE];

    for (auto _ : state) 
    {
        __x86_64_AVX_SSE_dest_aligned_memcpy(dst, src, BUFFER_SIZE);
        benchmark::ClobberMemory(); // Ensure the compiler doesn't optimize away the memcpy
    }
}
BENCHMARK(BM__x86_64_AVX_SSE_dest_aligned_memcpy);

static void BM__x86_64_AVX_SSE_source_aligned_memcpy(benchmark::State& state) 
{
    // Buffers
    alignas(32) char src[BUFFER_SIZE];
    alignas(32) char dst[BUFFER_SIZE];

    for (auto _ : state) 
    {
        __x86_64_AVX_SSE_source_aligned_memcpy(dst, src, BUFFER_SIZE);
        benchmark::ClobberMemory(); // Ensure the compiler doesn't optimize away the memcpy
    }
}
BENCHMARK(BM__x86_64_AVX_SSE_source_aligned_memcpy);




TEST(memcpy, __x86_64_AVX_SSE_aligned_memcpy)
{
    for (int i = 110; 0 <= i; --i)
    {
        // Buffers
        alignas(32) char src[128] = "https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE_ALL,AVX_ALL&ig_expand=5740";
        alignas(32) char dst[128];

        __x86_64_AVX_SSE_aligned_memcpy(dst, src, i);
        EXPECT_TRUE( std::memcmp(dst, src, i) == 0 );
        std::memset(dst, 0, sizeof(dst));
    }
}

TEST(memcpy, BM__x86_64_AVX_SSE_unaligned_memcpy)
{
    for (int i = 110; 0 <= i; --i)
    {
        // Buffers
        alignas(32) char src[128] = "https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE_ALL,AVX_ALL&ig_expand=5740";
        alignas(32) char dst[128];

        __x86_64_AVX_SSE_unaligned_memcpy(dst, src, i);
        EXPECT_TRUE( std::memcmp(dst, src, i) == 0 );
        std::memset(dst, 0, sizeof(dst));
    }
}

TEST(memcpy, BM__x86_64_AVX_SSE_dest_aligned_memcpy)
{
    for (int i = 110; 0 <= i; --i)
    {
        // Buffers
        alignas(32) char src[128] = "https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE_ALL,AVX_ALL&ig_expand=5740";
        alignas(32) char dst[128];

        __x86_64_AVX_SSE_dest_aligned_memcpy(dst, src, i);
        EXPECT_TRUE( std::memcmp(dst, src, i) == 0 );
        std::memset(dst, 0, sizeof(dst));
    }
}

TEST(memcpy, BM__x86_64_AVX_SSE_source_aligned_memcpy)
{
    for (int i = 110; 0 <= i; --i)
    {
        // Buffers
        alignas(32) char src[128] = "https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE_ALL,AVX_ALL&ig_expand=5740";
        alignas(32) char dst[128];

        __x86_64_AVX_SSE_source_aligned_memcpy(dst, src, i);
        EXPECT_TRUE( std::memcmp(dst, src, i) == 0 );
        std::memset(dst, 0, sizeof(dst));
    }
}




int main(int argc, char** argv) 
{   
	testing::InitGoogleTest(&argc, argv);
    
	if (argv == nullptr)
	{
		char arg0_default[] = "benchmark";
		char* args_default = arg0_default;
    	argc = 1;
		argv = &args_default;
	}

	benchmark::Initialize(&argc, argv);

	if (benchmark::ReportUnrecognizedArguments(argc, argv) == true)
    {
        std::cerr << "Failed to meet the expectation: Unrecognized Benchmark Arguments Detected.";
        return -1;
    } 

    int exit_code = RUN_ALL_TESTS();
	std::cerr << "\n\n";
	benchmark::RunSpecifiedBenchmarks();

    benchmark::Shutdown();                                            
    return exit_code;                                                           
}   

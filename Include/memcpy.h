#ifndef _SIMD_MEMORY_ALGORITHMS_PLATFORM_MEMCPY_H_
#define _SIMD_MEMORY_ALGORITHMS_PLATFORM_MEMCPY_H_
/*
Copyright © from 2022 to present, UNKNOWN STRYKER. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifdef __x86_64_AVX_SSE_aligned_memcpy
    #error __x86_64_AVX_SSE_aligned_memcpy is a function identifier.
#endif
#ifdef __x86_64_AVX_SSE_dest_aligned_memcpy
    #error __x86_64_AVX_SSE_dest_aligned_memcpy is a function identifier.
#endif
#ifdef __x86_64_AVX_SSE_source_aligned_memcpy
    #error __x86_64_AVX_SSE_source_aligned_memcpy is a function identifier.
#endif
#ifdef __x86_64_AVX_SSE_unaligned_memcpy
    #error __x86_64_AVX_SSE_unaligned_memcpy is a function identifier.
#endif

#ifdef __x86_64_AVX512_AVX_SSE_aligned_memcpy
    #error __x86_64_AVX512_AVX_SSE_aligned_memcpy is a function identifier.
#endif
#ifdef __x86_64_AVX512_AVX_SSE_dest_aligned_memcpy
    #error __x86_64_AVX512_AVX_SSE_dest_aligned_memcpy is a function identifier.
#endif
#ifdef __x86_64_AVX512_AVX_SSE_source_aligned_memcpy
    #error __x86_64_AVX512_AVX_SSE_source_aligned_memcpy is a function identifier.
#endif
#ifdef __x86_64_AVX512_AVX_SSE_unaligned_memcpy
    #error __x86_64_AVX512_AVX_SSE_unaligned_memcpy is a function identifier.
#endif

#include "platform.h"

#include <assert.h>
#include <stddef.h>
#include <immintrin.h>




#ifndef _SIMD_MEMORY_ALGORITHMS_FORCE_USING_C_IMPL_
    #ifdef _SIMD_MEMORY_ALGORITHMS_PLATFORM_X86_64_
    extern "C" void __x86_64_AVX_SSE_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);
    extern "C" void __x86_64_AVX_SSE_dest_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);
    extern "C" void __x86_64_AVX_SSE_source_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);
    extern "C" void __x86_64_AVX_SSE_unaligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);

    // extern void __x86_64_AVX512_AVX_SSE_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);
    // extern void __x86_64_AVX512_AVX_SSE_dest_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);
    // extern void __x86_64_AVX512_AVX_SSE_source_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);
    // extern void __x86_64_AVX512_AVX_SSE_unaligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);
    #endif
#else
    #ifdef _SIMD_MEMORY_ALGORITHMS_PLATFORM_X86_64_
    inline static void __x86_64_AVX_SSE_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p)
    {
	    assert(out_dest_p != nullptr && "Assertion failed: the out returning destination pointer is a null pointer.");
	    assert(out_dest_p != nullptr && "Assertion failed: the source pointer is a null pointer.");

	    for (__m256i* const end = static_cast<__m256i*>(out_dest_p) + (bytes_to_copy_p >> 5); out_dest_p != end;)
	    {
		    _mm256_store_si256(static_cast<__m256i*>(out_dest_p), _mm256_load_si256(static_cast<const __m256i*>(source_p)));
		    out_dest_p = static_cast<__m256i*>(out_dest_p) + 1;
		    source_p = static_cast<const __m256i*>(source_p) + 1;
	    }

	    bytes_to_copy_p = bytes_to_copy_p % 32;
        //if (bytes_to_copy_p >= 16)
        //{
        //	_mm_store_si128(static_cast<__m128i*>(out_dest_p), _mm_load_si128(static_cast<const __m128i*>(source_p)));
        //	out_dest_p = static_cast<__m128i*>(out_dest_p) + 1;
        //	source_p = static_cast<const __m128i*>(source_p) + 1;
        //	bytes_to_copy_p -= 16;
        //}

        for (int8_t* const end = static_cast<int8_t*>(out_dest_p) + bytes_to_copy_p; out_dest_p != end;)
        {
            *static_cast<int8_t*>(out_dest_p) = *static_cast<const int8_t*>(source_p);
            out_dest_p = static_cast<int8_t*>(out_dest_p) + 1;
            source_p = static_cast<const int8_t*>(source_p) + 1;
        }
    }

    inline static void __x86_64_AVX_SSE_dest_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p)
    {
	    assert(out_dest_p != nullptr && "Assertion failed: the out returning destination pointer is a null pointer.");
	    assert(out_dest_p != nullptr && "Assertion failed: the source pointer is a null pointer.");

	    for (__m256i* const end = static_cast<__m256i*>(out_dest_p) + (bytes_to_copy_p >> 5); out_dest_p != end;)
	    {
		    _mm256_store_si256(static_cast<__m256i*>(out_dest_p), _mm256_loadu_si256(static_cast<const __m256i*>(source_p)));
		    out_dest_p = static_cast<__m256i*>(out_dest_p) + 1;
		    source_p = static_cast<const __m256i*>(source_p) + 1;
	    }

	    bytes_to_copy_p = bytes_to_copy_p % 32;
        //if (bytes_to_copy_p >= 16)
        //{
        //	_mm_store_si128(static_cast<__m128i*>(out_dest_p), _mm_loadu_si128(static_cast<const __m128i*>(source_p)));
        //	out_dest_p = static_cast<__m128i*>(out_dest_p) + 1;
        //	source_p = static_cast<const __m128i*>(source_p) + 1;
        //	bytes_to_copy_p -= 16;
        //}

        for (int8_t* const end = static_cast<int8_t*>(out_dest_p) + bytes_to_copy_p; out_dest_p != end;)
        {
            *static_cast<int8_t*>(out_dest_p) = *static_cast<const int8_t*>(source_p);
            out_dest_p = static_cast<int8_t*>(out_dest_p) + 1;
            source_p = static_cast<const int8_t*>(source_p) + 1;
        }
    }

    inline static void __x86_64_AVX_SSE_source_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p)
    {
	    assert(out_dest_p != nullptr && "Assertion failed: the out returning destination pointer is a null pointer.");
	    assert(out_dest_p != nullptr && "Assertion failed: the source pointer is a null pointer.");

	    for (__m256i* const end = static_cast<__m256i*>(out_dest_p) + (bytes_to_copy_p >> 5); out_dest_p != end;)
	    {
		    _mm256_storeu_si256(static_cast<__m256i*>(out_dest_p), _mm256_load_si256(static_cast<const __m256i*>(source_p)));
		    out_dest_p = static_cast<__m256i*>(out_dest_p) + 1;
		    source_p = static_cast<const __m256i*>(source_p) + 1;
	    }

	    bytes_to_copy_p = bytes_to_copy_p % 32;
        //if (bytes_to_copy_p >= 16)
        //{
        //	_mm_storeu_si128(static_cast<__m128i*>(out_dest_p), _mm_load_si128(static_cast<const __m128i*>(source_p)));
        //	out_dest_p = static_cast<__m128i*>(out_dest_p) + 1;
        //	source_p = static_cast<const __m128i*>(source_p) + 1;
        //	bytes_to_copy_p -= 16;
        //}

        for (int8_t* const end = static_cast<int8_t*>(out_dest_p) + bytes_to_copy_p; out_dest_p != end;)
        {
            *static_cast<int8_t*>(out_dest_p) = *static_cast<const int8_t*>(source_p);
            out_dest_p = static_cast<int8_t*>(out_dest_p) + 1;
            source_p = static_cast<const int8_t*>(source_p) + 1;
        }
    }

    inline static void __x86_64_AVX_SSE_unaligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p)
    {
	    assert(out_dest_p != nullptr && "Assertion failed: the out returning destination pointer is a null pointer.");
	    assert(out_dest_p != nullptr && "Assertion failed: the source pointer is a null pointer.");

	    for (__m256i* const end = static_cast<__m256i*>(out_dest_p) + (bytes_to_copy_p >> 5); out_dest_p != end;)
	    {
		    _mm256_storeu_si256(static_cast<__m256i*>(out_dest_p), _mm256_loadu_si256(static_cast<const __m256i*>(source_p)));
		    out_dest_p = static_cast<__m256i*>(out_dest_p) + 1;
		    source_p = static_cast<const __m256i*>(source_p) + 1;
	    }

	    bytes_to_copy_p = bytes_to_copy_p % 32;
        //if (bytes_to_copy_p >= 16)
        //{
        //	_mm_storeu_si128(static_cast<__m128i*>(out_dest_p), _mm_loadu_si128(static_cast<const __m128i*>(source_p)));
        //	out_dest_p = static_cast<__m128i*>(out_dest_p) + 1;
        //	source_p = static_cast<const __m128i*>(source_p) + 1;
        //	bytes_to_copy_p -= 16;
        //}

        for (int8_t* const end = static_cast<int8_t*>(out_dest_p) + bytes_to_copy_p; out_dest_p != end;)
        {
            *static_cast<int8_t*>(out_dest_p) = *static_cast<const int8_t*>(source_p);
            out_dest_p = static_cast<int8_t*>(out_dest_p) + 1;
            source_p = static_cast<const int8_t*>(source_p) + 1;
        }
    }

    // inline void __x86_64_AVX512_AVX_SSE_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);
    // inline void __x86_64_AVX512_AVX_SSE_dest_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);
    // inline void __x86_64_AVX512_AVX_SSE_source_aligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);
    // inline void __x86_64_AVX512_AVX_SSE_unaligned_memcpy(void* out_dest_p, const void* source_p, size_t bytes_to_copy_p);
    #endif
#endif
#endif
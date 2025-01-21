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

#include <cstddef>
#include <immintrin.h>




extern "C"
{
//#ifdef _SIMD_MEMORY_ALGORITHMS_PLATFORM_X86_64_
    void __x86_64_AVX_SSE_aligned_memcpy(void* dest_p, const void* source_p, size_t bytes_to_copy_p);
    void __x86_64_AVX_SSE_dest_aligned_memcpy(void* dest_p, const void* source_p, size_t bytes_to_copy_p);
    void __x86_64_AVX_SSE_source_aligned_memcpy(void* dest_p, const void* source_p, size_t bytes_to_copy_p);
    void __x86_64_AVX_SSE_unaligned_memcpy(void* dest_p, const void* source_p, size_t bytes_to_copy_p);

    // void __x86_64_AVX512_AVX_SSE_aligned_memcpy(void* dest_p, const void* source_p, size_t bytes_to_copy_p);
    // void __x86_64_AVX512_AVX_SSE_dest_aligned_memcpy(void* dest_p, const void* source_p, size_t bytes_to_copy_p);
    // void __x86_64_AVX512_AVX_SSE_source_aligned_memcpy(void* dest_p, const void* source_p, size_t bytes_to_copy_p);
    // void __x86_64_AVX512_AVX_SSE_unaligned_memcpy(void* dest_p, const void* source_p, size_t bytes_to_copy_p);
//#endif
}


#endif
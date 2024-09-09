#ifndef _FE_CORE_MEMORY_HXX_
#define _FE_CORE_MEMORY_HXX_
// Copyright © 2023~ UNKNOWN STRYKER. All Rights Reserved.

#ifdef FE_UNALIGNED_MEMSET
	#error FE_UNALIGNED_MEMSET is a reserved Frogman Engine macro keyword.
#endif
#ifdef FE_ALIGNED_MEMSET
	#error FE_ALIGNED_MEMSET is a reserved Frogman Engine macro keyword.
#endif
#ifdef FE_UNALIGNED_MEMCPY
	#error FE_UNALIGNED_MEMCPY is a reserved Frogman Engine macro keyword.
#endif
#ifdef FE_ALIGNED_MEMCPY
	#error FE_ALIGNED_MEMCPY is a reserved Frogman Engine macro keyword.
#endif
#ifdef FE_DEST_ALIGNED_MEMCPY
	#error FE_DEST_ALIGNED_MEMCPY is a reserved Frogman Engine macro keyword.
#endif
#ifdef FE_SOURCE_ALIGNED_MEMCPY
	#error FE_SOURCE_ALIGNED_MEMCPY is a reserved Frogman Engine macro keyword.
#endif
#ifdef FE_ALIGNED_MEMMOVE
	#error FE_ALIGNED_MEMMOVE is a reserved Frogman Engine macro keyword.
#endif
#ifdef FE_UNALIGNED_MEMMOVE
	#error FE_UNALIGNED_MEMMOVE is a reserved Frogman Engine macro keyword.
#endif


#include <FE/prerequisites.h>
#include <FE/algorithm/math.hpp>


#ifdef _X86_64_
	// AVX, AVX 512, _mm_malloc, and _mm_free
	#include <immintrin.h>

	// SSE2 intrinsics
	#include <emmintrin.h>

	#ifdef __SSE2__
		#define _SSE2_
	#endif

	#ifdef __AVX__
		#define _AVX_
	#endif

	#ifdef __AVX2__
		#define _AVX2_
	#endif

	#ifdef __AVX512F__
		#define _AVX512F_
	#endif

#elif defined(_ARM64_)
	#if defined(__ARM_NEON) || defined(__ARM_NEON__)
		#define _ARM_NEON_
		#include <arm_neon.h>
	#endif

	#ifdef __ARM_NEON_FP
		#define _ARM_NEON_FP_
		#include <arm_neon.h>
	#endif
#endif




BEGIN_NAMESPACE(FE)


_MAYBE_UNUSED_ constexpr uint8 byte_size = 1;
_MAYBE_UNUSED_ constexpr uint8 word_size = 2;
_MAYBE_UNUSED_ constexpr uint8 dword_size = 4;
_MAYBE_UNUSED_ constexpr uint8 qword_size = 8;

using reserve = size;
using resize_to = size;
using extend = size;

struct align_8bytes final
{
	_MAYBE_UNUSED_ static constexpr size size = 8;
};

struct align_16bytes final
{
	_MAYBE_UNUSED_ static constexpr size size = 16;
};

struct align_32bytes final
{
	_MAYBE_UNUSED_ static constexpr size size = 32;
};

struct align_64bytes final
{
	_MAYBE_UNUSED_ static constexpr size size = 64;
};

struct align_128bytes final
{
	_MAYBE_UNUSED_ static constexpr size size = 128;
};

struct align_CPU_L1_cache_line final
{
    /*
    	Sixty-four bytes is commonly used CPU L1 cache line size for X86-64, but the actual size may vary.
    	A hardcoded value is used if the current system's compiler does not support std::hardware_destructive_interference_size.
	*/
	#ifdef _CLANG_
	_MAYBE_UNUSED_ static constexpr size size = 64;
	#else
    _MAYBE_UNUSED_ static constexpr size size = std::hardware_destructive_interference_size;
	#endif
};

template<uint64 PaddingSize>
struct align_custom_bytes final
{
	_MAYBE_UNUSED_ static constexpr inline uint64 size = PaddingSize;
};

struct SIMD_auto_alignment
{
#ifdef _AVX512F_
	using alignment_type = align_64bytes;
#elif defined(_AVX_) || defined(_AVX2_)
	using alignment_type = align_32bytes;
#else
	using alignment_type = align_16bytes;
#endif

	_MAYBE_UNUSED_ static constexpr size size = alignment_type::size;
};

#pragma warning(push)
#pragma warning(disable:4324)
template<typename T, class Alignment = typename FE::SIMD_auto_alignment>
struct alignas(Alignment::size) aligned final
{
	using value_type = T;
	using alignment_type = Alignment;

	T _data;
};
#pragma warning(pop)

enum struct ADDRESS : boolean
{
	_NOT_ALIGNED = false,
	_ALIGNED = true
};

#if defined(_AVX512F_) && defined(_AVX_) && defined(_SSE2_)
_FORCE_INLINE_ void __x86_64_unaligned_memset_AVX512_AVX_SSE2(void* out_dest_p, int8 value_p, size bytes_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));

	size l_leftover_bytes = MODULO_BY_64(bytes_p);
	// FE_DIVIDE_BY_64(bytes_p) == SIMD operation count
	for (__m512i* const end = static_cast<__m512i*>(out_dest_p) + FE_DIVIDE_BY_64(bytes_p); out_dest_p != end;)
	{
		_mm512_storeu_si512(static_cast<__m512i*>(out_dest_p), _mm512_set1_epi8(value_p));
		out_dest_p = static_cast<__m512i*>(out_dest_p) + 1;
	}

    if(l_leftover_bytes > 0)
	{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    	std::memset(out_dest_p, value_p, l_leftover_bytes);
#else
	    __x86_64_unaligned_memset_AVX_SSE2(out_dest_p, value_p, l_leftover_bytes);
#endif
		return;
	}
}

_FORCE_INLINE_ void __x86_64_aligned_memset_AVX512_AVX_SSE2(void* out_dest_p, int8 value_p, size bytes_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(MODULO_BY_64(reinterpret_cast<uintptr>(out_dest_p)) != 0, "${%s@0}: The address is not aligned by 64.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_ILLEGAL_ADDRESS_ALIGNMENT));

	size l_leftover_bytes = MODULO_BY_64(bytes_p);
	// FE_DIVIDE_BY_64(bytes_p) == SIMD operation count
	for (__m512i* const end = static_cast<__m512i*>(out_dest_p) + FE_DIVIDE_BY_64(bytes_p); out_dest_p != end;)
	{
		_mm512_store_si512(static_cast<__m512i*>(out_dest_p), _mm512_set1_epi8(value_p));
		out_dest_p = static_cast<__m512i*>(out_dest_p) + 1;
	}

    if(l_leftover_bytes > 0)
	{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    	std::memset(out_dest_p, value_p, l_leftover_bytes);
#else
	    __x86_64_aligned_memset_AVX_SSE2(out_dest_p, value_p, l_leftover_bytes);
#endif
		return;
	}
}


_FORCE_INLINE_ void __x86_64_unaligned_memcpy_AVX512_AVX_SSE2(void* out_dest_p, const void* source_p, size bytes_to_copy_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}

	size l_leftover_bytes = MODULO_BY_64(bytes_to_copy_p);
	// FE_DIVIDE_BY_64(bytes_to_copy_p) == SIMD operation count
	for (__m512i* const end = static_cast<__m512i*>(out_dest_p) + FE_DIVIDE_BY_64(bytes_to_copy_p); out_dest_p != end;)
	{
		_mm512_storeu_si512(static_cast<__m512i*>(out_dest_p), _mm512_loadu_si512(static_cast<const __m512i*>(source_p)));
		out_dest_p = static_cast<__m512i*>(out_dest_p) + 1;
		source_p = static_cast<const __m512i*>(source_p) + 1;
	}

    if(l_leftover_bytes > 0)
	{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    	std::memcpy(out_dest_p, source_p, l_leftover_bytes);
#else
	    __x86_64_unaligned_memcpy_AVX_SSE2(out_dest_p, source_p, l_leftover_bytes);
#endif
		return;
	}
}

_FORCE_INLINE_ void __x86_64_aligned_memcpy_AVX512_AVX_SSE2(void* out_dest_p, const void* source_p, size bytes_to_copy_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	FE_ASSERT(FE_MODULO_BY_64(reinterpret_cast<uintptr>(out_dest_p)) != 0, "${%s@}: out_dest_p is not aligned by 64.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_ILLEGAL_ADDRESS_ALIGNMENT));
	FE_ASSERT(FE_MODULO_BY_64(reinterpret_cast<uintptr>(source_p)) != 0, "${%s@}: source_p is not aligned by 64.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_ILLEGAL_ADDRESS_ALIGNMENT));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}

	size l_leftover_bytes = FE_MODULO_BY_64(bytes_to_copy_p);
	// FE_DIVIDE_BY_64(bytes_to_copy_p) == SIMD operation count
	for (__m512i* const end = static_cast<__m512i*>(out_dest_p) + FE_DIVIDE_BY_64(bytes_to_copy_p); out_dest_p != end;)
	{
		_mm512_store_si512(static_cast<__m512i*>(out_dest_p), _mm512_load_si512(static_cast<const __m512i*>(source_p)));
		out_dest_p = static_cast<__m512i*>(out_dest_p) + 1;
		source_p = static_cast<const __m512i*>(source_p) + 1;
	}

    if(l_leftover_bytes > 0)
	{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    	std::memcpy(out_dest_p, source_p, l_leftover_bytes);
#else
	    __x86_64_aligned_memcpy_AVX_SSE2(out_dest_p, source_p, l_leftover_bytes);
#endif
		return;
	}
}

_FORCE_INLINE_ void __x86_64_dest_aligned_memcpy_AVX512_AVX_SSE2(void* out_dest_p, const void* source_p, size bytes_to_copy_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	FE_ASSERT(FE_MODULO_BY_64(reinterpret_cast<uintptr>(out_dest_p)) != 0, "${%s@}: out_dest_p is not aligned by 64.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_ILLEGAL_ADDRESS_ALIGNMENT));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}

	size l_leftover_bytes = FE_MODULO_BY_64(bytes_to_copy_p);
	// FE_DIVIDE_BY_64(bytes_to_copy_p) == SIMD operation count
	for (__m512i* const end = static_cast<__m512i*>(out_dest_p) + FE_DIVIDE_BY_64(bytes_to_copy_p); out_dest_p != end;)
	{
		_mm512_store_si512(static_cast<__m512i*>(out_dest_p), _mm512_loadu_si512(static_cast<const __m512i*>(source_p)));
		out_dest_p = static_cast<__m512i*>(out_dest_p) + 1;
		source_p = static_cast<const __m512i*>(source_p) + 1;
	}

    if(l_leftover_bytes > 0)
	{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    	std::memcpy(out_dest_p, source_p, l_leftover_bytes);
#else
	    __x86_64_dest_aligned_memcpy_AVX_SSE2(out_dest_p, source_p, l_leftover_bytes);
#endif
		return;
	}
}

_FORCE_INLINE_ void __x86_64_source_aligned_memcpy_AVX512_AVX_SSE2(void* out_dest_p, const void* source_p, size bytes_to_copy_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	FE_ASSERT(FE_MODULO_BY_64(reinterpret_cast<uintptr>(source_p)) != 0, "${%s@}: source_p is not aligned by 64.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_ILLEGAL_ADDRESS_ALIGNMENT));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}

	size l_leftover_bytes = FE_MODULO_BY_64(bytes_to_copy_p);
	// FE_DIVIDE_BY_64(bytes_to_copy_p) == SIMD operation count
	for (__m512i* const end = static_cast<__m512i*>(out_dest_p) + FE_DIVIDE_BY_64(bytes_to_copy_p); out_dest_p != end;)
	{
		_mm512_storeu_si512(static_cast<__m512i*>(out_dest_p), _mm512_load_si512(static_cast<const __m512i*>(source_p)));
		out_dest_p = static_cast<__m512i*>(out_dest_p) + 1;
		source_p = static_cast<const __m512i*>(source_p) + 1;
	}

    if(l_leftover_bytes > 0)
	{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    	std::memcpy(out_dest_p, source_p, l_leftover_bytes);
#else
	    __x86_64_source_aligned_memcpy_AVX_SSE2(out_dest_p, source_p, l_leftover_bytes);
#endif
		return;
	}
}


_FORCE_INLINE_ void __x86_64_unaligned_memmove_AVX512_AVX_SSE2(void* const out_dest_p, const void* const source_p, size bytes_to_move_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	FE_ASSERT(bytes_to_move_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(bytes_to_move_p));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}

	if (source_p < out_dest_p)
	{
		var::byte* l_dest_byte_ptr = static_cast<var::byte*>(out_dest_p) + (bytes_to_move_p - 1);
		byte* l_source_byte_ptr = static_cast<byte*>(source_p) + (bytes_to_move_p - 1);

		{
			size l_leftover_bytes_to_copy_by_byte = FE_MODULO_BY_64(bytes_to_move_p);

			for (var::size i = 0; i != l_leftover_bytes_to_copy_by_byte; ++i)
			{
				*l_dest_byte_ptr = *l_source_byte_ptr;
				--l_dest_byte_ptr;
				--l_source_byte_ptr;
			}
		}

		var::size l_operation_count = FE_DIVIDE_BY_64(bytes_to_move_p);

		__m512i* l_m512i_dest_ptr = reinterpret_cast<__m512i*>(l_dest_byte_ptr - 63);
		const __m512i* l_m512i_source_ptr = reinterpret_cast<const __m512i*>(l_source_byte_ptr - 63);

		for (; l_operation_count > 1; --l_operation_count)
		{
			_mm512_storeu_si512(l_m512i_dest_ptr, _mm512_loadu_si512(l_m512i_source_ptr));
			--l_m512i_dest_ptr;
			--l_m512i_source_ptr;
		}

		size l_leftover_bytes = reinterpret_cast<var::byte*>(l_m512i_dest_ptr) - static_cast<var::byte* const>(out_dest_p);

		if (l_leftover_bytes > 0)
		{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    		std::memmove(out_dest_p, source_p, l_leftover_bytes);
#else
			__x86_64_unaligned_memmove_AVX_SSE2(out_dest_p, source_p, l_leftover_bytes);
#endif
			return;
		}
		return;
	}
	else
	{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    	std::memcpy(out_dest_p, source_p, bytes_to_move_p);
#else
		__x86_64_unaligned_memcpy_AVX512_AVX_SSE2(out_dest_p, source_p, bytes_to_move_p);
#endif
		return;
	}
}

_FORCE_INLINE_ void __x86_64_aligned_memmove_AVX512_AVX_SSE2(void* const out_dest_p, const void* const source_p, size bytes_to_move_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	FE_ASSERT(bytes_to_move_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(bytes_to_move_p));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}

	if (source_p < out_dest_p)
	{
		var::byte* l_dest_byte_ptr = static_cast<var::byte*>(out_dest_p) + (bytes_to_move_p - 1);
		byte* l_source_byte_ptr = static_cast<byte*>(source_p) + (bytes_to_move_p - 1);

		{
			size l_leftover_bytes_to_copy_by_byte = FE_MODULO_BY_64(bytes_to_move_p);

			for (var::size i = 0; i != l_leftover_bytes_to_copy_by_byte; ++i)
			{
				*l_dest_byte_ptr = *l_source_byte_ptr;
				--l_dest_byte_ptr;
				--l_source_byte_ptr;
			}
		}

		var::size l_operation_count = FE_DIVIDE_BY_64(bytes_to_move_p);

		__m512i* l_m512i_dest_ptr = reinterpret_cast<__m512i*>(l_dest_byte_ptr - 63);
		const __m512i* l_m512i_source_ptr = reinterpret_cast<const __m512i*>(l_source_byte_ptr - 63);

		for (; l_operation_count > 1; --l_operation_count)
		{
			_mm512_store_si512(l_m512i_dest_ptr, _mm512_loadu_si512(l_m512i_source_ptr));
			--l_m512i_dest_ptr;
			--l_m512i_source_ptr;
		}

		size l_leftover_bytes = reinterpret_cast<var::byte*>(l_m512i_dest_ptr) - static_cast<var::byte* const>(out_dest_p);

		if (l_leftover_bytes > 0)
		{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    		std::memmove(out_dest_p, source_p, l_leftover_bytes);
#else
			__x86_64_aligned_memmove_AVX_SSE2(out_dest_p, source_p, l_leftover_bytes);
#endif
			return;
		}
		return;
	}
	else
	{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    	std::memcpy(out_dest_p, source_p, bytes_to_move_p);
#else
		__x86_64_dest_aligned_memcpy_AVX512_AVX_SSE2(out_dest_p, source_p, bytes_to_move_p);
#endif
		return;
	}
}

#elif defined(_AVX_) && defined(_SSE2_)
_FORCE_INLINE_ void __x86_64_unaligned_memset_AVX_SSE2(void* out_dest_p, int8 value_p, size bytes_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));

	var::size l_leftover_bytes = FE_MODULO_BY_32(bytes_p);

	// FE_DIVIDE_BY_32(bytes_p) == SIMD operation count
	for (__m256i* const end = static_cast<__m256i*>(out_dest_p) + FE_DIVIDE_BY_32(bytes_p); out_dest_p != end;)
	{
		_mm256_storeu_si256(static_cast<__m256i*>(out_dest_p), _mm256_set1_epi8(value_p));
		out_dest_p = static_cast<__m256i*>(out_dest_p) + 1;
	}

	if (l_leftover_bytes >= 16)
	{
		_mm_storeu_si128(static_cast<__m128i*>(out_dest_p), _mm_set1_epi8(value_p));
		out_dest_p = static_cast<__m128i*>(out_dest_p) + 1;
		l_leftover_bytes -= 16;
	}

	for (var::byte* const end = static_cast<var::byte*>(out_dest_p) + l_leftover_bytes; out_dest_p != end;)
	{
		*static_cast<var::byte*>(out_dest_p) = static_cast<var::byte>(value_p);
		out_dest_p = static_cast<var::byte*>(out_dest_p) + 1;
	}
}

_FORCE_INLINE_ void __x86_64_aligned_memset_AVX_SSE2(void* out_dest_p, int8 value_p, size bytes_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(FE_MODULO_BY_32(reinterpret_cast<uintptr>(out_dest_p)) != 0, "${%s@0}: The address is not aligned by 32.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_ILLEGAL_ADDRESS_ALIGNMENT));

	var::size l_leftover_bytes = FE_MODULO_BY_32(bytes_p);

	// FE_DIVIDE_BY_32(bytes_p) == SIMD operation count
	for (__m256i* const end = static_cast<__m256i*>(out_dest_p) + FE_DIVIDE_BY_32(bytes_p); out_dest_p != end;)
	{
		_mm256_store_si256(static_cast<__m256i*>(out_dest_p), _mm256_set1_epi8(value_p));
		out_dest_p = static_cast<__m256i*>(out_dest_p) + 1;
	}

	if (l_leftover_bytes >= 16)
	{
		_mm_store_si128(static_cast<__m128i*>(out_dest_p), _mm_set1_epi8(value_p));
		out_dest_p = static_cast<__m128i*>(out_dest_p) + 1;
		l_leftover_bytes -= 16;
	}

	for (var::byte* const end = static_cast<var::byte*>(out_dest_p) + l_leftover_bytes; out_dest_p != end;)
	{
		*static_cast<var::byte*>(out_dest_p) = static_cast<var::byte>(value_p);
		out_dest_p = static_cast<var::byte*>(out_dest_p) + 1;
	}
}


_FORCE_INLINE_ void __x86_64_unaligned_memcpy_AVX_SSE2(void* out_dest_p, const void* source_p, size bytes_to_copy_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}

	var::size l_leftover_bytes = FE_MODULO_BY_32(bytes_to_copy_p);
	
	// FE_DIVIDE_BY_32(bytes_to_copy_p) == SIMD operation count
	for (__m256i* const end = static_cast<__m256i*>(out_dest_p) + FE_DIVIDE_BY_32(bytes_to_copy_p); out_dest_p != end;)
	{
		_mm256_storeu_si256(static_cast<__m256i*>(out_dest_p), _mm256_loadu_si256(static_cast<const __m256i*>(source_p)));
		out_dest_p = static_cast<__m256i*>(out_dest_p) + 1;
		source_p = static_cast<const __m256i*>(source_p) + 1;
	}

	if (l_leftover_bytes >= 16)
	{
		_mm_storeu_si128(static_cast<__m128i*>(out_dest_p), _mm_loadu_si128(static_cast<const __m128i*>(source_p)));
		out_dest_p = static_cast<__m128i*>(out_dest_p) + 1;
		source_p = static_cast<const __m128i*>(source_p) + 1;
		l_leftover_bytes -= 16;
	}

	for (var::byte* const end = static_cast<var::byte*>(out_dest_p) + l_leftover_bytes; out_dest_p != end;)
	{
		*static_cast<var::byte*>(out_dest_p) = *static_cast<byte*>(source_p);
		out_dest_p = static_cast<var::byte*>(out_dest_p) + 1;
		source_p = static_cast<byte*>(source_p) + 1;
	}
}

_FORCE_INLINE_ void __x86_64_aligned_memcpy_AVX_SSE2(void* out_dest_p, const void* source_p, size bytes_to_copy_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	FE_ASSERT(FE_MODULO_BY_32(reinterpret_cast<uintptr>(out_dest_p)) != 0, "${%s@0}: ${%s@1} is not aligned by 32.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_ILLEGAL_ADDRESS_ALIGNMENT), TO_STRING(out_dest_p));
	FE_ASSERT(FE_MODULO_BY_32(reinterpret_cast<uintptr>(source_p)) != 0, "${%s@0}: ${%s@1} is not aligned by 32.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_ILLEGAL_ADDRESS_ALIGNMENT), TO_STRING(source_p));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}

	var::size l_leftover_bytes = FE_MODULO_BY_32(bytes_to_copy_p);

	// FE_DIVIDE_BY_32(bytes_to_copy_p) == SIMD operation count
	for (__m256i* const end = static_cast<__m256i*>(out_dest_p) + FE_DIVIDE_BY_32(bytes_to_copy_p); out_dest_p != end;)
	{
		_mm256_store_si256(static_cast<__m256i*>(out_dest_p), _mm256_load_si256(static_cast<const __m256i*>(source_p)));
		out_dest_p = static_cast<__m256i*>(out_dest_p) + 1;
		source_p = static_cast<const __m256i*>(source_p) + 1;
	}

	if (l_leftover_bytes >= 16)
	{
		_mm_store_si128(static_cast<__m128i*>(out_dest_p), _mm_load_si128(static_cast<const __m128i*>(source_p)));
		out_dest_p = static_cast<__m128i*>(out_dest_p) + 1;
		source_p = static_cast<const __m128i*>(source_p) + 1;
		l_leftover_bytes -= 16;
	}

	for (var::byte* const end = static_cast<var::byte*>(out_dest_p) + l_leftover_bytes; out_dest_p != end;)
	{
		*static_cast<var::byte*>(out_dest_p) = *static_cast<byte*>(source_p);
		out_dest_p = static_cast<var::byte*>(out_dest_p) + 1;
		source_p = static_cast<byte*>(source_p) + 1;
	}
}

_FORCE_INLINE_ void __x86_64_dest_aligned_memcpy_AVX_SSE2(void* out_dest_p, const void* source_p, size bytes_to_copy_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	FE_ASSERT(FE_MODULO_BY_32(reinterpret_cast<uintptr>(out_dest_p)) != 0, "${%s@0}: ${%s@1} is not aligned by 32.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_ILLEGAL_ADDRESS_ALIGNMENT), TO_STRING(out_dest_p));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}

	var::size l_leftover_bytes = FE_MODULO_BY_32(bytes_to_copy_p);

	// FE_DIVIDE_BY_32(bytes_to_copy_p) == SIMD operation count
	for (__m256i* const end = static_cast<__m256i*>(out_dest_p) + FE_DIVIDE_BY_32(bytes_to_copy_p); out_dest_p != end;)
	{
		_mm256_store_si256(static_cast<__m256i*>(out_dest_p), _mm256_loadu_si256(static_cast<const __m256i*>(source_p)));
		out_dest_p = static_cast<__m256i*>(out_dest_p) + 1;
		source_p = static_cast<const __m256i*>(source_p) + 1;
	}

	if (l_leftover_bytes >= 16)
	{
		_mm_store_si128(static_cast<__m128i*>(out_dest_p), _mm_loadu_si128(static_cast<const __m128i*>(source_p)));
		out_dest_p = static_cast<__m128i*>(out_dest_p) + 1;
		source_p = static_cast<const __m128i*>(source_p) + 1;
		l_leftover_bytes -= 16;
	}

	for (var::byte* const end = static_cast<var::byte*>(out_dest_p) + l_leftover_bytes; out_dest_p != end;)
	{
		*static_cast<var::byte*>(out_dest_p) = *static_cast<byte*>(source_p);
		out_dest_p = static_cast<var::byte*>(out_dest_p) + 1;
		source_p = static_cast<byte*>(source_p) + 1;
	}
}

_FORCE_INLINE_ void __x86_64_source_aligned_memcpy_AVX_SSE2(void* out_dest_p, const void* source_p, size bytes_to_copy_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	FE_ASSERT(FE_MODULO_BY_32(reinterpret_cast<uintptr>(source_p)) != 0, "${%s@0}: ${%s@1} is not aligned by 32.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_ILLEGAL_ADDRESS_ALIGNMENT), TO_STRING(source_p));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}

	var::size l_leftover_bytes = FE_MODULO_BY_32(bytes_to_copy_p);

	// FE_DIVIDE_BY_32(bytes_to_copy_p) == SIMD operation count
	for (__m256i* const end = static_cast<__m256i*>(out_dest_p) + FE_DIVIDE_BY_32(bytes_to_copy_p); out_dest_p != end;)
	{
		_mm256_storeu_si256(static_cast<__m256i*>(out_dest_p), _mm256_load_si256(static_cast<const __m256i*>(source_p)));
		
		out_dest_p = static_cast<__m256i*>(out_dest_p) + 1;
		source_p = static_cast<const __m256i*>(source_p) + 1;
	}

	if (l_leftover_bytes >= 16)
	{
		_mm_storeu_si128(static_cast<__m128i*>(out_dest_p), _mm_load_si128(static_cast<const __m128i*>(source_p)));
		out_dest_p = static_cast<__m128i*>(out_dest_p) + 1;
		source_p = static_cast<const __m128i*>(source_p) + 1;
		l_leftover_bytes -= 16;
	}

	for (var::byte* const end = static_cast<var::byte*>(out_dest_p) + l_leftover_bytes; out_dest_p != end;)
	{
		*static_cast<var::byte*>(out_dest_p) = *static_cast<byte*>(source_p);
		out_dest_p = static_cast<var::byte*>(out_dest_p) + 1;
		source_p = static_cast<byte*>(source_p) + 1;
	}
}

/*
if (source_p < out_dest_p) being true means that the two void*s "out_dest_p and source_p" are pointing to the same range of memory
possibly overlap each other. If that is the case
memmove iterates and copies the data in the reverse order of memcpy operation.

        copy & traversal order 
    <---------------------------
  front                       back
    ++++++++++++++++++++++++++++
    |                          |
    ++++++++++++++++++++++++++++
low address               high address

Otherwise, it invokes memcpy to proceed its operation.
*/
_FORCE_INLINE_ void __x86_64_unaligned_memmove_AVX_SSE2(void* out_dest_p, const void* source_p, size bytes_to_move_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	FE_ASSERT(bytes_to_move_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(bytes_to_move_p));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}

	if (source_p < out_dest_p)
	{
		var::byte* l_dest_byte_ptr = static_cast<var::byte*>(out_dest_p) + (bytes_to_move_p - 1);
		byte* l_source_byte_ptr = static_cast<byte*>(source_p) + (bytes_to_move_p - 1);

		{
			size l_leftover_bytes_to_copy_by_byte = FE_MODULO_BY_16(bytes_to_move_p);

			for (var::size i = 0; i != l_leftover_bytes_to_copy_by_byte; ++i)
			{
				*l_dest_byte_ptr = *l_source_byte_ptr;
				--l_dest_byte_ptr;
				--l_source_byte_ptr;
			}
		}

		var::size l_operation_count = FE_DIVIDE_BY_16(bytes_to_move_p);

		__m256i* l_m256i_dest_ptr = reinterpret_cast<__m256i*>(l_dest_byte_ptr - 31);
		const __m256i* l_m256i_source_ptr = reinterpret_cast<const __m256i*>(l_source_byte_ptr  - 31);
	
		for (; l_operation_count > 1; l_operation_count -= 2)
		{
			_mm256_storeu_si256(l_m256i_dest_ptr, _mm256_loadu_si256(l_m256i_source_ptr));
			--l_m256i_dest_ptr;
			--l_m256i_source_ptr;
		}

		__m128i* l_m128i_dest_ptr = reinterpret_cast<__m128i*>(l_m256i_dest_ptr) + 1;
		const __m128i* l_m128i_source_ptr = reinterpret_cast<const __m128i*>(l_m256i_source_ptr) + 1;

		for (; l_operation_count > 0; --l_operation_count)
		{
			_mm_storeu_si128(l_m128i_dest_ptr, _mm_loadu_si128(l_m128i_source_ptr));
			--l_m128i_dest_ptr;
			--l_m128i_source_ptr;
		}
		return;
	}
	else
	{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    	std::memcpy(out_dest_p, source_p, bytes_to_move_p);
#else
		__x86_64_unaligned_memcpy_AVX_SSE2(out_dest_p, source_p, bytes_to_move_p);
#endif
		return;
	}
}

/*
if (source_p < out_dest_p) being true means that the two void*s "out_dest_p and source_p" are pointing to the same range of memory
possibly overlap each other. If that is the case
memmove iterates and copies the data in the reverse order of memcpy operation.

        copy & traversal order 
    <---------------------------
  front                       back
    ++++++++++++++++++++++++++++
    |                          |
    ++++++++++++++++++++++++++++
low address               high address

Otherwise, it invokes memcpy to proceed its operation.
*/
_FORCE_INLINE_ void __x86_64_aligned_memmove_AVX_SSE2(void* out_dest_p, const void* source_p, size bytes_to_move_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(source_p));
	FE_ASSERT(bytes_to_move_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_INVALID_SIZE), TO_STRING(bytes_to_move_p));
	if(UNLIKELY(out_dest_p == source_p)) _UNLIKELY_
	{
    	return;
	}
	FE_ASSERT(FE_MODULO_BY_32(reinterpret_cast<uintptr>(out_dest_p)) != 0, "${%s@0}: ${%s@1} is not aligned by 32.", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_ILLEGAL_ADDRESS_ALIGNMENT), TO_STRING(out_dest_p));

	if (source_p < out_dest_p)
	{
		var::byte* l_dest_byte_ptr = static_cast<var::byte*>(out_dest_p) + (bytes_to_move_p - 1);
		byte* l_source_byte_ptr = static_cast<byte*>(source_p) + (bytes_to_move_p - 1);

		{
			size l_leftover_bytes_to_copy_by_byte = FE_MODULO_BY_16(bytes_to_move_p);

			for (var::size i = 0; i != l_leftover_bytes_to_copy_by_byte; ++i)
			{
				*l_dest_byte_ptr = *l_source_byte_ptr;
				--l_dest_byte_ptr;
				--l_source_byte_ptr;
			}
		}

		var::size l_operation_count = FE_DIVIDE_BY_16(bytes_to_move_p);

		__m256i* l_m256i_dest_ptr = reinterpret_cast<__m256i*>(l_dest_byte_ptr - 31);
		const __m256i* l_m256i_source_ptr = reinterpret_cast<const __m256i*>(l_source_byte_ptr + 31);

		for (; l_operation_count > 1; l_operation_count -= 2)
		{
			_mm256_store_si256(l_m256i_dest_ptr, _mm256_loadu_si256(l_m256i_source_ptr));
			--l_m256i_dest_ptr;
			--l_m256i_source_ptr;
		}

		__m128i* l_m128i_dest_ptr = reinterpret_cast<__m128i*>(l_m256i_dest_ptr) + 1;
		const __m128i* l_m128i_source_ptr = reinterpret_cast<const __m128i*>(l_m256i_source_ptr) + 1;

		for (; l_operation_count > 0; --l_operation_count)
		{
			_mm_store_si128(l_m128i_dest_ptr, _mm_loadu_si128(l_m128i_source_ptr));
			--l_m128i_dest_ptr;
			--l_m128i_source_ptr;
		}
		return;
	}
	else
	{
#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    	std::memcpy(out_dest_p, source_p, bytes_to_move_p);
#else
		__x86_64_dest_aligned_memcpy_AVX_SSE2(out_dest_p, source_p, bytes_to_move_p);
#endif
		return;
	}
}
#endif


#if defined(_AVX512F_) && defined(_AVX_) && defined(_SSE2_)
	#define FE_UNALIGNED_MEMSET(out_dest_p, value_p, bytes_p) ::FE::__x86_64_unaligned_memset_AVX512_AVX_SSE2(out_dest_p, value_p, bytes_p)
	#define FE_ALIGNED_MEMSET(out_dest_p, value_p, bytes_p) ::FE::__x86_64_aligned_memset_AVX512_AVX_SSE2(out_dest_p, value_p, bytes_p)
	#define FE_UNALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::FE::__x86_64_unaligned_memcpy_AVX512_AVX_SSE2(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::FE::__x86_64_aligned_memcpy_AVX512_AVX_SSE2(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_DEST_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::FE::__x86_64_dest_aligned_memcpy_AVX512_AVX_SSE2(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_SOURCE_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::FE::__x86_64_source_aligned_memcpy_AVX512_AVX_SSE2(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_UNALIGNED_MEMMOVE(out_dest_p, source_p, bytes_to_move_p) ::FE::__x86_64_unaligned_memmove_AVX512_AVX_SSE2(out_dest_p, source_p, bytes_to_move_p)
	#define FE_ALIGNED_MEMMOVE(out_dest_p, source_p, bytes_to_move_p) ::FE::__x86_64_aligned_memmove_AVX512_AVX_SSE2(out_dest_p, source_p, bytes_to_move_p)
#elif defined(_AVX_) && defined(_SSE2_)
	#define FE_UNALIGNED_MEMSET(out_dest_p, value_p, bytes_p) ::FE::__x86_64_unaligned_memset_AVX_SSE2(out_dest_p, value_p, bytes_p)
	#define FE_ALIGNED_MEMSET(out_dest_p, value_p, bytes_p) ::FE::__x86_64_aligned_memset_AVX_SSE2(out_dest_p, value_p, bytes_p)
	#define FE_UNALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::FE::__x86_64_unaligned_memcpy_AVX_SSE2(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::FE::__x86_64_aligned_memcpy_AVX_SSE2(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_DEST_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::FE::__x86_64_dest_aligned_memcpy_AVX_SSE2(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_SOURCE_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::FE::__x86_64_source_aligned_memcpy_AVX_SSE2(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_UNALIGNED_MEMMOVE(out_dest_p, source_p, bytes_to_move_p) ::FE::__x86_64_unaligned_memmove_AVX_SSE2(out_dest_p, source_p, bytes_to_move_p)
	#define FE_ALIGNED_MEMMOVE(out_dest_p, source_p, bytes_to_move_p) ::FE::__x86_64_aligned_memmove_AVX_SSE2(out_dest_p, source_p, bytes_to_move_p)
#else
	#define FE_UNALIGNED_MEMSET(out_dest_p, value_p, bytes_p) ::std::memset(out_dest_p, value_p, bytes_p)
	#define FE_ALIGNED_MEMSET(out_dest_p, value_p, bytes_p) ::std::memset(out_dest_p, value_p, bytes_p)
	#define FE_UNALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::std::memcpy(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::std::memcpy(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_DEST_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::std::memcpy(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_SOURCE_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_to_copy_p) ::std::memcpy(out_dest_p, source_p, bytes_to_copy_p)
	#define FE_UNALIGNED_MEMMOVE(out_dest_p, source_p, bytes_to_move_p) ::std::memmove(out_dest_p, source_p, bytes_to_move_p)
	#define FE_ALIGNED_MEMMOVE(out_dest_p, source_p, bytes_to_move_p) ::std::memmove(out_dest_p, source_p, bytes_to_move_p)
#endif

template<typename T, class Alignment>
_FORCE_INLINE_ size calculate_aligned_memory_size_in_bytes(count_t elements_p) noexcept  
{
	FE_ASSERT(elements_p == 0, "Assertion Failure: ${%s@0} cannot be zero.", TO_STRING(elements_p));

	size l_actual_size = sizeof(T) * elements_p;
	var::size l_multiplier = l_actual_size / sizeof(FE::aligned<T, Alignment>);
	l_multiplier += ((l_actual_size % sizeof(FE::aligned<T, Alignment>)) != 0);

	return sizeof(FE::aligned<T, Alignment>) * l_multiplier;
}

template<class ConstIterator>
_REGISTER_CALL_ boolean memcmp(ConstIterator left_iterator_begin_p, ConstIterator left_iterator_end_p, ConstIterator right_iterator_begin_p, ConstIterator right_iterator_end_p) noexcept  
{
	static_assert(std::is_class<ConstIterator>::value == true);
	FE_ASSERT(left_iterator_begin_p == nullptr, "ERROR: left_iterator_begin_p is nullptr.");
	FE_ASSERT(left_iterator_end_p == nullptr, "ERROR: left_iterator_end_p is nullptr.");
	FE_ASSERT(right_iterator_begin_p == nullptr, "ERROR: right_iterator_begin_p is nullptr.");
	FE_ASSERT(right_iterator_end_p == nullptr, "ERROR: right_iterator_end_p is nullptr.");

	ConstIterator l_left_iterator_begin = left_iterator_begin_p;

	if ((left_iterator_end_p - left_iterator_begin_p) != (right_iterator_end_p - right_iterator_begin_p))
	{
		return false;
	}

	while ((l_left_iterator_begin != left_iterator_end_p) && (*l_left_iterator_begin == *right_iterator_begin_p))
	{
		++l_left_iterator_begin;
		++right_iterator_begin_p;
	}

	if ((l_left_iterator_begin - left_iterator_begin_p) == (left_iterator_end_p - left_iterator_begin_p))
	{
		return true;
	}

	return false;
}

template<ADDRESS DestAddressAlignment = ADDRESS::_NOT_ALIGNED, ADDRESS SourceAddressAlignment = ADDRESS::_NOT_ALIGNED>
_FORCE_INLINE_ void memcpy(void* out_dest_p, size dest_capacity_in_bytes_p, const void* source_p, count_t source_capacity_in_bytes_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_NULLPTR), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_NULLPTR), TO_STRING(source_p));

#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    std::memcpy(out_dest_p, source_p, FE_MIN(dest_capacity_in_bytes_p, source_capacity_in_bytes_p));
#else
	if constexpr (DestAddressAlignment == ADDRESS::_ALIGNED && SourceAddressAlignment == ADDRESS::_ALIGNED)
	{
		FE_ALIGNED_MEMCPY(out_dest_p, source_p, FE_MIN(dest_capacity_in_bytes_p, source_capacity_in_bytes_p));
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_ALIGNED && SourceAddressAlignment == ADDRESS::_NOT_ALIGNED)
	{
		FE_DEST_ALIGNED_MEMCPY(out_dest_p, source_p, FE_MIN(dest_capacity_in_bytes_p, source_capacity_in_bytes_p));
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_NOT_ALIGNED && SourceAddressAlignment == ADDRESS::_ALIGNED)
	{
		FE_SOURCE_ALIGNED_MEMCPY(out_dest_p, source_p, FE_MIN(dest_capacity_in_bytes_p, source_capacity_in_bytes_p));
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_NOT_ALIGNED && SourceAddressAlignment == ADDRESS::_NOT_ALIGNED)
	{
		FE_UNALIGNED_MEMCPY(out_dest_p, source_p, FE_MIN(dest_capacity_in_bytes_p, source_capacity_in_bytes_p));
	}
#endif
}

template<ADDRESS DestAddressAlignment = ADDRESS::_NOT_ALIGNED, ADDRESS SourceAddressAlignment = ADDRESS::_NOT_ALIGNED>
_FORCE_INLINE_ void memcpy(void* out_dest_p, const void* source_p, count_t bytes_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_NULLPTR), TO_STRING(out_dest_p));
	FE_ASSERT(source_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_NULLPTR), TO_STRING(source_p));

#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    std::memcpy(out_dest_p, source_p, bytes_p);
#else
	if constexpr (DestAddressAlignment == ADDRESS::_ALIGNED && SourceAddressAlignment == ADDRESS::_ALIGNED)
	{
		FE_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_p);
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_ALIGNED && SourceAddressAlignment == ADDRESS::_NOT_ALIGNED)
	{
		FE_DEST_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_p);
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_NOT_ALIGNED && SourceAddressAlignment == ADDRESS::_ALIGNED)
	{
		FE_SOURCE_ALIGNED_MEMCPY(out_dest_p, source_p, bytes_p);
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_NOT_ALIGNED && SourceAddressAlignment == ADDRESS::_NOT_ALIGNED)
	{
		FE_UNALIGNED_MEMCPY(out_dest_p, source_p, bytes_p);
	}
#endif
}

template<ADDRESS DestAddressAlignment = ADDRESS::_NOT_ALIGNED>
_FORCE_INLINE_ void memset(void* out_dest_p, int8 value_p, count_t bytes_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_NULLPTR), TO_STRING(out_dest_p));

#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    std::memset(out_dest_p, value_p, bytes_p);
#else
	if constexpr (DestAddressAlignment == ADDRESS::_ALIGNED)
	{
		FE_ALIGNED_MEMSET(out_dest_p, (int8)value_p, bytes_p);
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_NOT_ALIGNED)
	{
		FE_UNALIGNED_MEMSET(out_dest_p, (int8)value_p, bytes_p);
	}
#endif
}

template<ADDRESS DestAddressAlignment = ADDRESS::_NOT_ALIGNED>
_FORCE_INLINE_ void memmove(void* out_dest_p, const void* source_p, size bytes_p) noexcept  
{
	FE_ASSERT(out_dest_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_NULLPTR), TO_STRING(out_dest_p));
	FE_ASSERT(bytes_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(FE::ERROR_CODE::_FATAL_MEMORY_ERROR_1XX_NULLPTR), TO_STRING(bytes_p));

#if defined(_DEBUG_) && !defined(_RELWITHDEBINFO_)
    std::memmove(out_dest_p, source_p, bytes_p);
#else
	if constexpr (DestAddressAlignment == ADDRESS::_ALIGNED)
	{
		FE_ALIGNED_MEMMOVE(out_dest_p, source_p, bytes_p);
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_NOT_ALIGNED)
	{
		FE_UNALIGNED_MEMMOVE(out_dest_p, source_p, bytes_p);
	}
#endif
}


END_NAMESPACE
#endif

#ifndef _FE_CORE_MEMORY_HXX_
#define _FE_CORE_MEMORY_HXX_
// Copyright © 2023~ UNKNOWN STRYKER. All Rights Reserved.
#ifdef UNALIGNED_MEMSET
#error UNALIGNED_MEMSET is a reserved Frogman Engine macro keyword.
#endif
#ifdef ALIGNED_MEMSET
#error ALIGNED_MEMSET is a reserved Frogman Engine macro keyword.
#endif
#ifdef UNALIGNED_MEMCPY
#error UNALIGNED_MEMCPY is a reserved Frogman Engine macro keyword.
#endif
#ifdef ALIGNED_MEMCPY
#error ALIGNED_MEMCPY is a reserved Frogman Engine macro keyword.
#endif
#ifdef DEST_ALIGNED_MEMCPY
#error DEST_ALIGNED_MEMCPY is a reserved Frogman Engine macro keyword.
#endif
#ifdef SOURCE_ALIGNED_MEMCPY
#error SOURCE_ALIGNED_MEMCPY is a reserved Frogman Engine macro keyword.
#endif


#ifdef __AVX__
#define _AVX_
#endif

#ifdef __AVX512F__
#define _AVX512_
#endif

#include <immintrin.h>
#include <FE/core/prerequisites.h>
#include <FE/core/algorithm/math.h>




BEGIN_NAMESPACE(FE)


enum struct OBJECT_STATUS : boolean
{
	_CONSTRUCTED = true,
	_DESTRUCTED = false
};


#ifdef _AVX512_
_FORCE_INLINE_ void unaligned_memset_with_avx512(void* const out_dest_pointer_p, int8 value_p, size_t total_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(total_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(total_bytes_p));

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(out_dest_pointer_p);
	const __m512i l_m512i_value_to_be_assigned = _mm512_set1_epi8(value_p);

	var::size_t l_leftover_bytes = MODULO_BY_64(total_bytes_p);
	size_t l_avx512_operation_count = MODULO_BY_64(total_bytes_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx512_operation_count; ++executed_operation_count)
	{
		_mm512_storeu_si512(l_m512i_dest_ptr, l_m512i_value_to_be_assigned);
		++l_m512i_dest_ptr;
	}

	if (l_leftover_bytes >= 16)
	{
		memset(l_m512i_dest_ptr, value_p, l_leftover_bytes);
		return;
	}

	var::byte* l_byte_ptr = reinterpret_cast<var::byte*>(l_m512i_dest_ptr);
	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_byte_ptr = value_p;
		++l_byte_ptr;
	}
}

_FORCE_INLINE_ void aligned_memset_with_avx512(void* const out_dest_pointer_p, int8 value_p, size_t total_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(total_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(total_bytes_p));
	FE_ASSERT(MODULO_BY_64(reinterpret_cast<uintptr_t>(out_dest_pointer_p)) != 0, "${%s@0}: The address is not aligned by 64.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT));

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(out_dest_pointer_p);
	const __m512i l_m512i_value_to_be_assigned = _mm512_set1_epi8(value_p);

	var::size_t l_leftover_bytes = MODULO_BY_64(total_bytes_p);
	size_t l_avx512_operation_count = MODULO_BY_64(total_bytes_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx512_operation_count; ++executed_operation_count)
	{
		_mm512_store_si512(l_m512i_dest_ptr, l_m512i_value_to_be_assigned);
		++l_m512i_dest_ptr;
	}

	if (l_leftover_bytes >= 16)
	{
		memset(l_m512i_dest_ptr, value_p, l_leftover_bytes);
		return;
	}

	var::byte* l_byte_ptr = reinterpret_cast<var::byte*>(l_m512i_dest_ptr);
	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_byte_ptr = value_p;
		++l_byte_ptr;
	}
}


_FORCE_INLINE_ void unaligned_memcpy_with_avx512(void* const out_dest_pointer_p, const void* const source_pointer_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_pointer_p));
	FE_ASSERT(bytes_to_copy_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(bytes_to_copy_p));

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(out_dest_pointer_p);
	const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_64(bytes_to_copy_p);
	size_t l_avx512_operation_count = MODULO_BY_64(bytes_to_copy_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx512_operation_count; ++executed_operation_count)
	{
		_mm512_storeu_si512(l_m512i_dest_ptr, _mm512_loadu_si512(l_m512i_source_ptr));
		++l_m512i_dest_ptr;
		++l_m512i_source_ptr;
	}

	if (l_leftover_bytes >= 16)
	{
		memcpy(l_m512i_dest_ptr, l_m512i_source_ptr, l_leftover_bytes);
		return;
	}

	var::byte* l_dest_byte_ptr = reinterpret_cast<var::byte*>(l_m512i_dest_ptr);
	byte* l_source_byte_ptr = reinterpret_cast<byte*>(l_m512i_source_ptr);
	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_dest_byte_ptr = *l_source_byte_ptr;
		++l_dest_byte_ptr;
		++l_source_byte_ptr;
	}
}

_FORCE_INLINE_ void aligned_memcpy_with_avx512(void* const out_dest_pointer_p, const void* const source_pointer_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_pointer_p));
	FE_ASSERT(bytes_to_copy_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(bytes_to_copy_p));
	FE_ASSERT(MODULO_BY_64(reinterpret_cast<uintptr_t>(out_dest_pointer_p)) != 0, "${%s@}: out_dest_pointer_p is not aligned by 64.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT));
	FE_ASSERT(MODULO_BY_64(reinterpret_cast<uintptr_t>(source_pointer_p)) != 0, "${%s@}: source_pointer_p is not aligned by 64.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT));

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(out_dest_pointer_p);
	const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_64(bytes_to_copy_p);
	size_t l_avx512_operation_count = MODULO_BY_64(bytes_to_copy_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx512_operation_count; ++executed_operation_count)
	{
		_mm512_store_si512(l_m512i_dest_ptr, _mm512_load_si512(l_m512i_source_ptr));
		++l_m512i_dest_ptr;
		++l_m512i_source_ptr;
	}

	if (l_leftover_bytes >= 16)
	{
		memcpy(l_m512i_dest_ptr, l_m512i_source_ptr, l_leftover_bytes);
		return;
	}

	var::byte* l_dest_byte_ptr = reinterpret_cast<var::byte*>(l_m512i_dest_ptr);
	byte* l_source_byte_ptr = reinterpret_cast<byte*>(l_m512i_source_ptr);
	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_dest_byte_ptr = *l_source_byte_ptr;
		++l_dest_byte_ptr;
		++l_source_byte_ptr;
	}
}

_FORCE_INLINE_ void dest_aligned_memcpy_with_avx512(void* const out_dest_pointer_p, const void* const source_pointer_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_pointer_p));
	FE_ASSERT(bytes_to_copy_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(bytes_to_copy_p));
	FE_ASSERT(MODULO_BY_64(reinterpret_cast<uintptr_t>(out_dest_pointer_p)) != 0, "${%s@}: out_dest_pointer_p is not aligned by 64.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT));

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(out_dest_pointer_p);
	const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_64(bytes_to_copy_p);
	size_t l_avx512_operation_count = MODULO_BY_64(bytes_to_copy_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx512_operation_count; ++executed_operation_count)
	{
		_mm512_store_si512(l_m512i_dest_ptr, _mm512_loadu_si512(l_m512i_source_ptr));
		++l_m512i_dest_ptr;
		++l_m512i_source_ptr;
	}

	if (l_leftover_bytes >= 16)
	{
		memcpy(l_m512i_dest_ptr, l_m512i_source_ptr, l_leftover_bytes);
		return;
	}

	var::byte* l_dest_byte_ptr = reinterpret_cast<var::byte*>(l_m512i_dest_ptr);
	byte* l_source_byte_ptr = reinterpret_cast<byte*>(l_m512i_source_ptr);
	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_dest_byte_ptr = *l_source_byte_ptr;
		++l_dest_byte_ptr;
		++l_source_byte_ptr;
	}
}

_FORCE_INLINE_ void source_aligned_memcpy_with_avx512(void* const out_dest_pointer_p, const void* const source_pointer_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_pointer_p));
	FE_ASSERT(bytes_to_copy_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(bytes_to_copy_p));
	FE_ASSERT(MODULO_BY_64(reinterpret_cast<uintptr_t>(source_pointer_p)) != 0, "${%s@}: source_pointer_p is not aligned by 64.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT));

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(out_dest_pointer_p);
	const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_64(bytes_to_copy_p);
	size_t l_avx512_operation_count = MODULO_BY_64(bytes_to_copy_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx512_operation_count; ++executed_operation_count)
	{
		_mm512_storeu_si512(l_m512i_dest_ptr, _mm512_load_si512(l_m512i_source_ptr));
		++l_m512i_dest_ptr;
		++l_m512i_source_ptr;
	}

	if (l_leftover_bytes >= 16)
	{
		memcpy(l_m512i_dest_ptr, l_m512i_source_ptr, l_leftover_bytes);
		return;
	}

	var::byte* l_dest_byte_ptr = reinterpret_cast<var::byte*>(l_m512i_dest_ptr);
	byte* l_source_byte_ptr = reinterpret_cast<byte*>(l_m512i_source_ptr);
	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_dest_byte_ptr = *l_source_byte_ptr;
		++l_dest_byte_ptr;
		++l_source_byte_ptr;
	}
}
#elif defined(_AVX_)
_FORCE_INLINE_ void unaligned_memset_with_avx(void* const out_dest_pointer_p, int8 value_p, size_t total_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(total_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(total_bytes_p));

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(out_dest_pointer_p);
	const __m256i l_m256i_value_to_be_assigned = _mm256_set1_epi8(value_p);

	var::size_t l_leftover_bytes = MODULO_BY_32(total_bytes_p);
	size_t l_avx_operation_count = DIVIDE_BY_32(total_bytes_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
	{
		_mm256_storeu_si256(l_m256i_dest_ptr, l_m256i_value_to_be_assigned);
		++l_m256i_dest_ptr;
	}

	__m128i* l_m128i_dest_ptr = reinterpret_cast<__m128i*>(l_m256i_dest_ptr);
	if (l_leftover_bytes >= 16)
	{
		_mm_storeu_si128(l_m128i_dest_ptr, _mm_set1_epi8(value_p));
		++l_m128i_dest_ptr;
		l_leftover_bytes -= 16;
	}
	var::byte* l_byte_ptr = reinterpret_cast<var::byte*>(l_m128i_dest_ptr);

	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_byte_ptr = value_p;
		++l_byte_ptr;
	}
}

_FORCE_INLINE_ void aligned_memset_with_avx(void* const out_dest_pointer_p, int8 value_p, size_t total_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(total_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(total_bytes_p));
	FE_ASSERT(MODULO_BY_32(reinterpret_cast<uintptr_t>(out_dest_pointer_p)) != 0, "${%s@0}: The address is not aligned by 32.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT));

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(out_dest_pointer_p);
	const __m256i l_m256i_value_to_be_assigned = _mm256_set1_epi8(value_p);

	var::size_t l_leftover_bytes = MODULO_BY_32(total_bytes_p);
	size_t l_avx_operation_count = DIVIDE_BY_32(total_bytes_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
	{
		_mm256_store_si256(l_m256i_dest_ptr, l_m256i_value_to_be_assigned);
		++l_m256i_dest_ptr;
	}

	__m128i* l_m128i_dest_ptr = reinterpret_cast<__m128i*>(l_m256i_dest_ptr);
	if (l_leftover_bytes >= 16)
	{
		_mm_store_si128(l_m128i_dest_ptr, _mm_set1_epi8(value_p));
		++l_m128i_dest_ptr;
		l_leftover_bytes -= 16;
	}
	var::byte* l_byte_ptr = reinterpret_cast<var::byte*>(l_m128i_dest_ptr);

	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_byte_ptr = value_p;
		++l_byte_ptr;
	}
}


_FORCE_INLINE_ void unaligned_memcpy_with_avx(void* const out_dest_pointer_p, const void* const source_pointer_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_pointer_p));
	FE_ASSERT(bytes_to_copy_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(bytes_to_copy_p));

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(out_dest_pointer_p);
	const __m256i* l_m256i_source_ptr = static_cast<const __m256i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
	size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
	{
		_mm256_storeu_si256(l_m256i_dest_ptr, _mm256_loadu_si256(l_m256i_source_ptr));
		++l_m256i_dest_ptr;
		++l_m256i_source_ptr;
	}

	__m128i* l_m128i_dest_ptr = reinterpret_cast<__m128i*>(l_m256i_dest_ptr);
	const __m128i* l_m128i_source_ptr = reinterpret_cast<const __m128i*>(l_m256i_source_ptr);
	if (l_leftover_bytes >= 16)
	{
		_mm_storeu_si128(l_m128i_dest_ptr, _mm_loadu_si128(l_m128i_source_ptr));
		++l_m128i_dest_ptr;
		++l_m128i_source_ptr;
		l_leftover_bytes -= 16;
	}

	var::byte* l_dest_byte_ptr = reinterpret_cast<var::byte*>(l_m128i_dest_ptr);
	byte* l_source_byte_ptr = reinterpret_cast<byte*>(l_m128i_source_ptr);
	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_dest_byte_ptr = *l_source_byte_ptr;
		++l_dest_byte_ptr;
		++l_source_byte_ptr;
	}
}

_FORCE_INLINE_ void aligned_memcpy_with_avx(void* const out_dest_pointer_p, const void* const source_pointer_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_pointer_p));
	FE_ASSERT(bytes_to_copy_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(bytes_to_copy_p));
	FE_ASSERT(MODULO_BY_32(reinterpret_cast<uintptr_t>(out_dest_pointer_p)) != 0, "${%s@0}: ${%s@1} is not aligned by 32.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(MODULO_BY_32(reinterpret_cast<uintptr_t>(source_pointer_p)) != 0, "${%s@0}: ${%s@1} is not aligned by 32.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT), TO_STRING(source_pointer_p));
	
	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(out_dest_pointer_p);
	const __m256i* l_m256i_source_ptr = static_cast<const __m256i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
	size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
	{
		_mm256_store_si256(l_m256i_dest_ptr, _mm256_load_si256(l_m256i_source_ptr));
		++l_m256i_dest_ptr;
		++l_m256i_source_ptr;
	}

	__m128i* l_m128i_dest_ptr = reinterpret_cast<__m128i*>(l_m256i_dest_ptr);
	const __m128i* l_m128i_source_ptr = reinterpret_cast<const __m128i*>(l_m256i_source_ptr);
	if (l_leftover_bytes >= 16)
	{
		_mm_store_si128(l_m128i_dest_ptr, _mm_load_si128(l_m128i_source_ptr));
		++l_m128i_dest_ptr;
		++l_m128i_source_ptr;
		l_leftover_bytes -= 16;
	}

	var::byte* l_dest_byte_ptr = reinterpret_cast<var::byte*>(l_m128i_dest_ptr);
	byte* l_source_byte_ptr = reinterpret_cast<byte*>(l_m128i_source_ptr);
	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_dest_byte_ptr = *l_source_byte_ptr;
		++l_dest_byte_ptr;
		++l_source_byte_ptr;
	}
}

_FORCE_INLINE_ void dest_aligned_memcpy_with_avx(void* const out_dest_pointer_p, const void* const source_pointer_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_pointer_p));
	FE_ASSERT(bytes_to_copy_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(bytes_to_copy_p));
	FE_ASSERT(MODULO_BY_32(reinterpret_cast<uintptr_t>(out_dest_pointer_p)) != 0, "${%s@0}: ${%s@1} is not aligned by 32.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT), TO_STRING(out_dest_pointer_p));
	//FE_ASSERT((reinterpret_cast<uintptr_t>(source_ptrc_p) % 32) != 0, "${%s@0}: source_ptrc_p is not aligned by 32.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT));

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(out_dest_pointer_p);
	const __m256i* l_m256i_source_ptr = static_cast<const __m256i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
	size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
	{
		_mm256_store_si256(l_m256i_dest_ptr, _mm256_loadu_si256(l_m256i_source_ptr));
		++l_m256i_dest_ptr;
		++l_m256i_source_ptr;
	}

	__m128i* l_m128i_dest_ptr = reinterpret_cast<__m128i*>(l_m256i_dest_ptr);
	const __m128i* l_m128i_source_ptr = reinterpret_cast<const __m128i*>(l_m256i_source_ptr);
	if (l_leftover_bytes >= 16)
	{
		_mm_store_si128(l_m128i_dest_ptr, _mm_loadu_si128(l_m128i_source_ptr));
		++l_m128i_dest_ptr;
		++l_m128i_source_ptr;
		l_leftover_bytes -= 16;
	}

	var::byte* l_dest_byte_ptr = reinterpret_cast<var::byte*>(l_m128i_dest_ptr);
	byte* l_source_byte_ptr = reinterpret_cast<byte*>(l_m128i_source_ptr);
	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_dest_byte_ptr = *l_source_byte_ptr;
		++l_dest_byte_ptr;
		++l_source_byte_ptr;
	}
}

_FORCE_INLINE_ void source_aligned_memcpy_with_avx(void* const out_dest_pointer_p, const void* const source_pointer_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_pointer_p));
	FE_ASSERT(bytes_to_copy_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(bytes_to_copy_p));
	FE_ASSERT(MODULO_BY_32(reinterpret_cast<uintptr_t>(source_pointer_p)) != 0, "${%s@0}: ${%s@1} is not aligned by 32.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT), TO_STRING(source_pointer_p));

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(out_dest_pointer_p);
	const __m256i* l_m256i_source_ptr = static_cast<const __m256i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
	size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p - l_leftover_bytes);

	for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
	{
		_mm256_storeu_si256(l_m256i_dest_ptr, _mm256_load_si256(l_m256i_source_ptr));
		++l_m256i_dest_ptr;
		++l_m256i_source_ptr;
	}

	__m128i* l_m128i_dest_ptr = reinterpret_cast<__m128i*>(l_m256i_dest_ptr);
	const __m128i* l_m128i_source_ptr = reinterpret_cast<const __m128i*>(l_m256i_source_ptr);
	if (l_leftover_bytes >= 16)
	{
		_mm_storeu_si128(l_m128i_dest_ptr, _mm_load_si128(l_m128i_source_ptr));
		++l_m128i_dest_ptr;
		++l_m128i_source_ptr;
		l_leftover_bytes -= 16;
	}

	var::byte* l_dest_byte_ptr = reinterpret_cast<var::byte*>(l_m128i_dest_ptr);
	byte* l_source_byte_ptr = reinterpret_cast<byte*>(l_m128i_source_ptr);
	for (var::size_t i = 0; i != l_leftover_bytes; ++i)
	{
		*l_dest_byte_ptr = *l_source_byte_ptr;
		++l_dest_byte_ptr;
		++l_source_byte_ptr;
	}
}
#endif


#ifdef _AVX512_
#define UNALIGNED_MEMSET(out_dest_pointer_p, value_p, total_bytes_p) ::FE::unaligned_memset_with_avx512(out_dest_pointer_p, value_p, total_bytes_p)
#define ALIGNED_MEMSET(out_dest_pointer_p, value_p, total_bytes_p) ::FE::aligned_memset_with_avx512(out_dest_pointer_p, value_p, total_bytes_p)
#define UNALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::unaligned_memcpy_with_avx512(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::aligned_memcpy_with_avx512(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define DEST_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::dest_aligned_memcpy_with_avx512(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define SOURCE_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::source_aligned_memcpy_with_avx512(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)

#elif defined(_AVX_)
#define UNALIGNED_MEMSET(out_dest_pointer_p, value_p, total_bytes_p) ::FE::unaligned_memset_with_avx(out_dest_pointer_p, value_p, total_bytes_p)
#define ALIGNED_MEMSET(out_dest_pointer_p, value_p, total_bytes_p) ::FE::aligned_memset_with_avx(out_dest_pointer_p, value_p, total_bytes_p)
#define UNALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::unaligned_memcpy_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::aligned_memcpy_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define DEST_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::dest_aligned_memcpy_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define SOURCE_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::source_aligned_memcpy_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)

#else
#define UNALIGNED_MEMSET(out_dest_pointer_p, value_p, total_bytes_p) ::std::memset(out_dest_pointer_p, value_p, total_bytes_p)
#define ALIGNED_MEMSET(out_dest_pointer_p, value_p, total_bytes_p) ::std::memset(out_dest_pointer_p, value_p, total_bytes_p)
#define UNALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::std::memcpy(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::std::memcpy(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define DEST_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::memcpy(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define SOURCE_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::memcpy(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#endif


_FORCE_INLINE_ void unaligned_memcpy(void* const out_dest_pointer_p, length_t dest_length_p, size_t dest_element_bytes_p, const void* const source_memblock_pointer_p, length_t source_length_p, size_t source_element_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_memblock_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));

	FE_ASSERT(dest_length_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(dest_element_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_element_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));

	size_t l_source_size = source_element_bytes_p * source_length_p;
	size_t l_dest_size = dest_element_bytes_p * dest_length_p;

	if (l_source_size >= l_dest_size)
	{
		UNALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, l_dest_size);
	}
	else
	{
		UNALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, l_source_size);
	}
}


_FORCE_INLINE_ void aligned_memcpy(void* const out_dest_pointer_p, length_t dest_length_p, size_t dest_element_bytes_p, const void* const source_memblock_pointer_p, length_t source_length_p, size_t source_element_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_memblock_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));

	FE_ASSERT(dest_length_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(dest_element_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_element_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));

	size_t l_source_size = source_element_bytes_p * source_length_p;
	size_t l_dest_size = dest_element_bytes_p * dest_length_p;

	if (l_source_size >= l_dest_size)
	{
		ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, l_dest_size);
	}
	else
	{
		ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, l_source_size);
	}
}


_FORCE_INLINE_ void dest_aligned_memcpy(void* const out_dest_pointer_p, length_t dest_length_p, size_t dest_element_bytes_p, const void* const source_memblock_pointer_p, length_t source_length_p, size_t source_element_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_memblock_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));

	FE_ASSERT(dest_length_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(dest_element_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_element_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));

	size_t l_source_size = source_element_bytes_p * source_length_p;
	size_t l_dest_size = dest_element_bytes_p * dest_length_p;

	if (l_source_size >= l_dest_size)
	{
		DEST_ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, l_dest_size);
	}
	else
	{
		DEST_ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, l_source_size);
	}
}

_FORCE_INLINE_ void source_aligned_memcpy(void* const out_dest_pointer_p, length_t dest_length_p, size_t dest_element_bytes_p, const void* const source_memblock_pointer_p, length_t source_length_p, size_t source_element_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_memblock_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));

	FE_ASSERT(dest_length_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(dest_element_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_element_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));

	size_t l_source_size = source_element_bytes_p * source_length_p;
	size_t l_dest_size = dest_element_bytes_p * dest_length_p;

	if (l_source_size >= l_dest_size)
	{
		SOURCE_ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, l_dest_size);
	}
	else
	{
		SOURCE_ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, l_source_size);
	}
}


_FORCE_INLINE_ void unaligned_memcpy(void* const out_dest_pointer_p, size_t dest_bytes_p, const void* const source_memblock_pointer_p, size_t source_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_memblock_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));

	FE_ASSERT(dest_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(dest_bytes_p));
	FE_ASSERT(source_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_bytes_p));

	if (source_bytes_p >= dest_bytes_p)
	{
		UNALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, dest_bytes_p);
	}
	else
	{
		UNALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, source_bytes_p);
	}
}


_FORCE_INLINE_ void aligned_memcpy(void* const out_dest_pointer_p, size_t dest_bytes_p, const void* const source_memblock_pointer_p, size_t source_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_memblock_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));

	FE_ASSERT(dest_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(dest_bytes_p));
	FE_ASSERT(source_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_bytes_p));

	if (source_bytes_p >= dest_bytes_p)
	{
		ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, dest_bytes_p);
	}
	else
	{
		ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, source_bytes_p);
	}
}


_FORCE_INLINE_ void dest_aligned_memcpy(void* const out_dest_pointer_p, size_t dest_bytes_p, const void* const source_memblock_pointer_p, size_t source_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_memblock_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));

	FE_ASSERT(dest_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(dest_bytes_p));
	FE_ASSERT(source_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_bytes_p));

	if (source_bytes_p >= dest_bytes_p)
	{
	  	DEST_ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, dest_bytes_p);
	}
	else
	{
		DEST_ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, source_bytes_p);
	}
}

_FORCE_INLINE_ void source_aligned_memcpy(void* const out_dest_pointer_p, size_t dest_bytes_p, const void* const source_memblock_pointer_p, size_t source_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_memblock_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));

	FE_ASSERT(dest_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(dest_bytes_p));
	FE_ASSERT(source_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_bytes_p));

	if (source_bytes_p >= dest_bytes_p)
	{
		SOURCE_ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, dest_bytes_p);
	}
	else
	{
		SOURCE_ALIGNED_MEMCPY(out_dest_pointer_p, source_memblock_pointer_p, source_bytes_p);
	}
}




enum struct MEMORY_SIZE_SCALABILITY : boolean
{
	_STATICALLY_SIZED = false,
	_DYNAMICALLY_SIZED = true
};

enum struct MEMORY_ERROR_1XX : int16
{
	_NONE = 0,
	_ERROR_ILLEGAL_ADDRESS_ALIGNMENT = 100,
	_FATAL_ERROR_INVALID_SIZE = 101,
	_FATAL_ERROR_ILLEGAL_POSITION = 102,
	_FATAL_ERROR_NULLPTR = 103,
	_FATAL_ERROR_OUT_OF_RANGE = 104,
	_FATAL_ERROR_OUT_OF_CAPACITY = 105,
	_FATAL_ERROR_ACCESS_VIOLATION = 106,

};

struct total_memory_utilization_data
{
	var::int64 _global_total_bytes = 0;
	var::int64 _thread_local_total_bytes = 0;

	var::int64 _global_total_bytes_by_type = 0;
	var::int64 _thread_local_total_bytes_by_type = 0;
};

struct global_memory_utilization
{
	var::int64 _global_total_bytes = 0;
	var::int64 _thread_local_total_bytes = 0;
};

struct type_memory_utilization
{
	var::int64 _global_total_bytes_by_type = 0;
	var::int64 _thread_local_total_bytes_by_type = 0;
};


_MAYBE_UNUSED_ constexpr uint8 _BYTE_SIZE_ = 1;
_MAYBE_UNUSED_ constexpr uint8 _WORD_SIZE_ = 2;
_MAYBE_UNUSED_ constexpr uint8 _DWORD_SIZE_ = 4;
_MAYBE_UNUSED_ constexpr uint8 _QWORD_SIZE_ = 8;


// it is used when reserving memory of Frogman Engine data containers.
struct reserve final
{
	uint64 _value = 0;
};

struct resize_to final
{
	uint64 _value = 0;
};

struct extend final
{
	uint64 _value = 0;
};

struct count final
{
	uint64 _value = 0;
};


struct align_4bytes final
{
	_MAYBE_UNUSED_ static constexpr uint16 size = 4;
};

struct align_8bytes final
{
	_MAYBE_UNUSED_ static constexpr uint16 size = 8;
};

struct align_16bytes final
{
	_MAYBE_UNUSED_ static constexpr uint16 size = 16;
};

struct align_32bytes final
{
	_MAYBE_UNUSED_ static constexpr uint16 size = 32;
};

struct align_64bytes final
{
	_MAYBE_UNUSED_ static constexpr uint16 size = 64;
};

struct align_128bytes final
{
	_MAYBE_UNUSED_ static constexpr uint16 size = 128;
};


// it contains memory padding size.
template<uint64 PaddingSize>
struct align_custom_bytes final
{
	_MAYBE_UNUSED_ static constexpr inline uint16 size = PaddingSize;
};


#ifdef _AVX512_
template<typename T, class Alignment = align_64bytes>
#elif defined(_AVX_)
template<typename T, class Alignment = align_32bytes>
#endif
struct alignas(Alignment::size) align
{
	T _data;
};


END_NAMESPACE




BEGIN_NAMESPACE(FE)


template<class Iterator>
_FORCE_INLINE_ boolean memcmp_s(Iterator left_iterator_begin_p, Iterator left_iterator_end_p, Iterator right_iterator_begin_p, Iterator right_iterator_end_p) noexcept
{
	static_assert(std::is_class<Iterator>::value == true);
	FE_ASSERT(left_iterator_begin_p == nullptr, "ERROR: left_iterator_begin_p is nullptr.");
	FE_ASSERT(left_iterator_end_p == nullptr, "ERROR: left_iterator_end_p is nullptr.");
	FE_ASSERT(right_iterator_begin_p == nullptr, "ERROR: right_iterator_begin_p is nullptr.");
	FE_ASSERT(right_iterator_end_p == nullptr, "ERROR: right_iterator_end_p is nullptr.");

	Iterator l_left_iterator_begin = left_iterator_begin_p;

	if ((left_iterator_end_p - left_iterator_begin_p) != (right_iterator_end_p - right_iterator_begin_p))
	{
		return false;
	}

	while ((*l_left_iterator_begin == *right_iterator_begin_p) && (l_left_iterator_begin != left_iterator_end_p))
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


template<typename T, TYPE_TRIVIALITY IsTrivial = FE::is_trivially_constructible_and_destructible<T>::value>
struct stack_memory;

template<typename T>
struct stack_memory<T, TYPE_TRIVIALITY::_TRIVIAL>
{
	_MAYBE_UNUSED_ static constexpr inline TYPE_TRIVIALITY is_trivially_constructible_and_destructible = TYPE_TRIVIALITY::_TRIVIAL;
	_MAYBE_UNUSED_ static constexpr inline boolean is_allocated_from_an_address_aligned_allocator = false;
};

template<typename T>
struct stack_memory<T, TYPE_TRIVIALITY::_NOT_TRIVIAL>
{
	_MAYBE_UNUSED_ static constexpr inline TYPE_TRIVIALITY is_trivially_constructible_and_destructible = TYPE_TRIVIALITY::_NOT_TRIVIAL;
	_MAYBE_UNUSED_ static constexpr inline boolean is_allocated_from_an_address_aligned_allocator = false;
};




template<typename T, class AllocatedFrom, TYPE_TRIVIALITY IsTrivial = AllocatedFrom::is_trivially_constructible_and_destructible>
class type_traits;

template<typename T, class AllocatedFrom>
class type_traits<T, AllocatedFrom, TYPE_TRIVIALITY::_TRIVIAL>
{
public:
	_MAYBE_UNUSED_ static constexpr inline TYPE_TRIVIALITY is_trivially_constructible_and_destructible = TYPE_TRIVIALITY::_TRIVIAL;


	_FORCE_INLINE_ static void construct(_MAYBE_UNUSED_ T& dest_ref_p) noexcept
	{
	}

	_FORCE_INLINE_ static void construct(T& dest_ref_p, T value_p) noexcept
	{
		dest_ref_p = value_p;
	}


	_FORCE_INLINE_ static void copy_construct(T* dest_ptr_p, count_t dest_count_p, T* source_ptr_p, count_t source_count_p) noexcept
	{
		if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == true)
		{
			FE::dest_aligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
		else if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == false)
		{
			FE::unaligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
	}
	
	_FORCE_INLINE_ static void copy_construct(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == true)
		{
			DEST_ALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
		else if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == false)
		{
			UNALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
	}


	_FORCE_INLINE_ static void move_construct(T* dest_ptr_p, count_t dest_count_p, T* source_ptr_p, count_t source_count_p) noexcept
	{
		if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == true)
		{
			FE::dest_aligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
		else if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == false)
		{
			FE::unaligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
	}

	_FORCE_INLINE_ static void move_construct(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == true)
		{
			DEST_ALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
		else if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == false)
		{
			UNALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
	}


	_FORCE_INLINE_ static void destruct(_MAYBE_UNUSED_ T& dest_ref_p) noexcept
	{
	}

	_FORCE_INLINE_ static void destruct(_MAYBE_UNUSED_ T* dest_begin_ptr_p, _MAYBE_UNUSED_ T* dest_end_ptr_p) noexcept
	{
	}


	_FORCE_INLINE_ static void copy_assign(T* dest_ptr_p, count_t dest_count_p, T* source_ptr_p, count_t source_count_p) noexcept
	{
		if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == true)
		{
			FE::dest_aligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
		else if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == false)
		{
			FE::unaligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
	}

	_FORCE_INLINE_ static void copy_assign(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == true)
		{
			DEST_ALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
		else if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == false)
		{
			UNALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
	}


	_FORCE_INLINE_ static void move_assign(T* dest_ptr_p, count_t dest_count_p, T* source_ptr_p, count_t source_count_p) noexcept
	{
		if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == true)
		{
			FE::dest_aligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
		else if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == false)
		{
			FE::unaligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
	}

	_FORCE_INLINE_ static void move_assign(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == true)
		{
			DEST_ALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
		else if constexpr (AllocatedFrom::is_allocated_from_an_address_aligned_allocator == false)
		{
			UNALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
	}
};


template<typename T, class AllocatedFrom>
class type_traits<T, AllocatedFrom, TYPE_TRIVIALITY::_NOT_TRIVIAL>
{
public:
	_MAYBE_UNUSED_ static constexpr inline TYPE_TRIVIALITY is_trivially_constructible_and_destructible = TYPE_TRIVIALITY::_NOT_TRIVIAL;


	_FORCE_INLINE_ static void construct(_MAYBE_UNUSED_ T& dest_ref_p) noexcept
	{
		static_assert(std::is_constructible<T>::value == true, "static assertion failed: The typename T must be copy constructible.");

		new(&dest_ref_p) T();
	}

	_FORCE_INLINE_ static void construct(T& dest_ref_p, T value_p) noexcept
	{
		static_assert(std::is_constructible<T>::value == true, "static assertion failed: The typename T must be copy constructible.");

		new(&dest_ref_p) T(std::move(value_p));
	}


	_FORCE_INLINE_ static void copy_construct(T* dest_ptr_p, count_t dest_count_p, T* source_ptr_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(dest_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(dest_ptr_p));
		FE_ASSERT(source_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(source_ptr_p));

		FE_ASSERT(dest_count_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(dest_count_p));
		FE_ASSERT(source_count_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(source_count_p));

		if (dest_count_p <= source_count_p)
		{
			T* const dest_end_ptrc = dest_ptr_p + dest_count_p;
			while (dest_ptr_p != dest_end_ptrc)
			{
				new(dest_ptr_p) T(*source_ptr_p);
				++dest_ptr_p;
				++source_ptr_p;
			}
			return;
		}
		else
		{
			T* const source_end_ptrc = source_ptr_p + source_count_p;
			while (source_ptr_p != source_end_ptrc)
			{
				new(dest_ptr_p) T(*source_ptr_p);
				++dest_ptr_p;
				++source_ptr_p;
			}
			return;
		}
	}

	_FORCE_INLINE_ static void copy_construct(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		FE_ASSERT(dest_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(dest_ptr_p));
		FE_ASSERT(source_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(source_ptr_p));

		FE_ASSERT(count_to_copy_or_move_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(count_to_copy_or_move_p));

		for (var::count_t i = 0; i < count_to_copy_or_move_p; ++i)
		{
			new(dest_ptr_p) T(*source_ptr_p);
			++dest_ptr_p;
			++source_ptr_p;
		}
	}


	_FORCE_INLINE_ static void move_construct(T* dest_ptr_p, count_t dest_count_p, T* source_ptr_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(dest_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(dest_ptr_p));
		FE_ASSERT(source_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(source_ptr_p));

		FE_ASSERT(dest_count_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(dest_count_p));
		FE_ASSERT(source_count_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(source_count_p));

		if (dest_count_p <= source_count_p)
		{
			T* const dest_end_ptrc = dest_ptr_p + dest_count_p;
			while (dest_ptr_p != dest_end_ptrc)
			{
				new(dest_ptr_p) T(std::move(*source_ptr_p));
				++dest_ptr_p;
				++source_ptr_p;
			}
			return;
		}
		else
		{
			T* const source_end_ptrc = source_ptr_p + source_count_p;
			while (source_ptr_p != source_end_ptrc)
			{
				new(dest_ptr_p) T(std::move(*source_ptr_p));
				++dest_ptr_p;
				++source_ptr_p;
			}
			return;
		}
	}

	_FORCE_INLINE_ static void move_construct(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		FE_ASSERT(dest_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(dest_ptr_p));
		FE_ASSERT(source_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(source_ptr_p));

		FE_ASSERT(count_to_copy_or_move_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(count_to_copy_or_move_p));

		for (var::count_t i = 0; i < count_to_copy_or_move_p; ++i)
		{
			new(dest_ptr_p) T(std::move(*source_ptr_p));
			++dest_ptr_p;
			++source_ptr_p;
		}
	}


	_FORCE_INLINE_ static void destruct(_MAYBE_UNUSED_ T& dest_ref_p) noexcept
	{
		static_assert(std::is_destructible<T>::value == true, "static assertion failed: The typename T must be destructible.");
		dest_ref_p.~T();
	}

	_FORCE_INLINE_ static void destruct(_MAYBE_UNUSED_ T* dest_begin_ptr_p, _MAYBE_UNUSED_ T* const dest_end_ptrc_p) noexcept
	{
		static_assert(std::is_destructible<T>::value == true, "static assertion failed: The typename T must be destructible.");

		while (dest_begin_ptr_p != dest_end_ptrc_p)
		{
			dest_begin_ptr_p->~T();
			++dest_begin_ptr_p;
		}
	}


	_FORCE_INLINE_ static void copy_assign(T* dest_ptr_p, count_t dest_count_p, T* source_ptr_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(dest_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(dest_ptr_p));
		FE_ASSERT(source_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(source_ptr_p));

		FE_ASSERT(dest_count_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(dest_count_p));
		FE_ASSERT(source_count_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(source_count_p));

		if (dest_count_p <= source_count_p)
		{
			T* const dest_end_ptrc = dest_ptr_p + dest_count_p;
			while (dest_ptr_p != dest_end_ptrc)
			{
				*dest_ptr_p = *source_ptr_p;
				++dest_ptr_p;
				++source_ptr_p;
			}
			return;
		}
		else
		{
			T* const source_end_ptrc = source_ptr_p + source_count_p;
			while (source_ptr_p != source_end_ptrc)
			{
				*dest_ptr_p = *source_ptr_p;
				++dest_ptr_p;
				++source_ptr_p;
			}
			return;
		}
	}

	_FORCE_INLINE_ static void copy_assign(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		FE_ASSERT(dest_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(dest_ptr_p));
		FE_ASSERT(source_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(source_ptr_p));

		FE_ASSERT(count_to_copy_or_move_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(count_to_copy_or_move_p));

		for (var::count_t i = 0; i < count_to_copy_or_move_p; ++i)
		{
			*dest_ptr_p = *source_ptr_p;
			++dest_ptr_p;
			++source_ptr_p;
		}
	}


	_FORCE_INLINE_ static void move_assign(T* dest_ptr_p, count_t dest_count_p, T* source_ptr_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(dest_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(dest_ptr_p));
		FE_ASSERT(source_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(source_ptr_p));

		FE_ASSERT(dest_count_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(dest_count_p));
		FE_ASSERT(source_count_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(source_count_p));

		if (dest_count_p <= source_count_p)
		{
			T* const dest_end_ptrc = dest_ptr_p + dest_count_p;
			while (dest_ptr_p != dest_end_ptrc)
			{
				*dest_ptr_p = std::move(*source_ptr_p);
				++dest_ptr_p;
				++source_ptr_p;
			}
			return;
		}
		else
		{
			T* const source_end_ptrc = source_ptr_p + source_count_p;
			while (source_ptr_p != source_end_ptrc)
			{
				*dest_ptr_p = std::move(*source_ptr_p);
				++dest_ptr_p;
				++source_ptr_p;
			}
			return;
		}
	}

	_FORCE_INLINE_ static void move_assign(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		FE_ASSERT(dest_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(dest_ptr_p));
		FE_ASSERT(source_ptr_p == nullptr, "MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR: ${%s@0} is nullptr", TO_STRING(source_ptr_p));

		FE_ASSERT(count_to_copy_or_move_p == 0, "MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(count_to_copy_or_move_p));

		for (var::count_t i = 0; i < count_to_copy_or_move_p; ++i)
		{
			*dest_ptr_p = std::move(*source_ptr_p);
			++dest_ptr_p;
			++source_ptr_p;
		}
	}
};


END_NAMESPACE
#endif

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
#define DIVIDE_BY_2(input_p) ((input_p) >> 1)
#define DIVIDE_BY_4(input_p) ((input_p) >> 2)
#define DIVIDE_BY_8(input_p) ((input_p) >> 3)
#define DIVIDE_BY_16(input_p) ((input_p) >> 4)
#define DIVIDE_BY_32(input_p) ((input_p) >> 5)
#define DIVIDE_BY_64(input_p) ((input_p) >> 6)
#define DIVIDE_BY_128(input_p) ((input_p) >> 7)

#define MODULO_BY_2(input_p) ((input_p) & 1)
#define MODULO_BY_4(input_p) ((input_p) & 3)
#define MODULO_BY_8(input_p) ((input_p) & 7)
#define MODULO_BY_16(input_p) ((input_p) & 15)
#define MODULO_BY_32(input_p) ((input_p) & 31)
#define MODULO_BY_64(input_p) ((input_p) & 63)
#define MODULO_BY_128(input_p) ((input_p) & 127)
#define ABS(x) ((x < 0) ? x * -1 : x)

#ifdef __AVX__
#define _AVX_
#endif

#ifdef __AVX512F__
#define _AVX512_
#endif

#include <immintrin.h>
#include <FE/core/prerequisites.h>




BEGIN_NAMESPACE(FE)


enum struct OBJECT_STATUS : boolean
{
	_CONSTRUCTED = true,
	_DESTRUCTED = false
};


#ifdef _AVX512_
_FORCE_INLINE_ void unaligned_memset_with_avx512(void* const dest_ptrc_p, int8 value_p, size_t total_bytes_p) noexcept
{
	FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
	FE_ASSERT(total_bytes_p == 0, "ERROR: element_bytes_p is 0.");

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
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

_FORCE_INLINE_ void aligned_memset_with_avx512(void* const dest_ptrc_p, int8 value_p, size_t total_bytes_p) noexcept
{
	FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
	FE_ASSERT(total_bytes_p == 0, "ERROR: element_bytes_p is 0.");
	FE_ASSERT((reinterpret_cast<uintptr_t>(dest_ptrc_p) % 64) != 0, "ERROR: dest_ptrc_p is not aligned by 64.");

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
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


_FORCE_INLINE_ void unaligned_memcpy_with_avx512(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
	FE_ASSERT(bytes_to_copy_p == 0, "ERROR: element_bytes_p is 0.");

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
	const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_ptrc_p);

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

_FORCE_INLINE_ void aligned_memcpy_with_avx512(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
	FE_ASSERT(bytes_to_copy_p == 0, "ERROR: element_bytes_p is 0.");
	FE_ASSERT((reinterpret_cast<uintptr_t>(dest_ptrc_p) % 64) != 0, "ERROR: dest_ptrc_p is not aligned by 64.");
	//FE_ASSERT((reinterpret_cast<uintptr_t>(source_ptrc_p) % 64) != 0, "ERROR: source_ptrc_p is not aligned by 64.");

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
	const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_ptrc_p);

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
#elif defined(_AVX_)
_FORCE_INLINE_ void unaligned_memset_with_avx(void* const dest_ptrc_p, int8 value_p, size_t total_bytes_p) noexcept
{
	FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
	FE_ASSERT(total_bytes_p == 0, "ERROR: element_bytes_p is 0.");

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(dest_ptrc_p);
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

_FORCE_INLINE_ void aligned_memset_with_avx(void* const dest_ptrc_p, int8 value_p, size_t total_bytes_p) noexcept
{
	FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
	FE_ASSERT(total_bytes_p == 0, "ERROR: element_bytes_p is 0.");
	FE_ASSERT((reinterpret_cast<uintptr_t>(dest_ptrc_p) % 32) != 0, "ERROR: dest_ptrc_p is not aligned by 32.");

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(dest_ptrc_p);
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


_FORCE_INLINE_ void unaligned_memcpy_with_avx(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
	FE_ASSERT(bytes_to_copy_p == 0, "ERROR: element_bytes_p is 0.");

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(dest_ptrc_p);
	const __m256i* l_m256i_source_ptr = static_cast<const __m256i*>(source_ptrc_p);

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

_FORCE_INLINE_ void aligned_memcpy_with_avx(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
{
	FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
	FE_ASSERT(bytes_to_copy_p == 0, "ERROR: element_bytes_p is 0.");
	FE_ASSERT((reinterpret_cast<uintptr_t>(dest_ptrc_p) % 32) != 0, "ERROR: dest_ptrc_p is not aligned by 32.");
	//FE_ASSERT((reinterpret_cast<uintptr_t>(source_ptrc_p) % 32) != 0, "ERROR: source_ptrc_p is not aligned by 32.");

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(dest_ptrc_p);
	const __m256i* l_m256i_source_ptr = static_cast<const __m256i*>(source_ptrc_p);

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
#endif


#ifdef _AVX512_
#define UNALIGNED_MEMSET(dest_ptrc_p, value_p, total_bytes_p) ::FE::unaligned_memset_with_avx512(dest_ptrc_p, value_p, total_bytes_p)
#define ALIGNED_MEMSET(dest_ptrc_p, value_p, total_bytes_p) ::FE::aligned_memset_with_avx512(dest_ptrc_p, value_p, total_bytes_p)
#define UNALIGNED_MEMCPY(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p) ::FE::unaligned_memcpy_with_avx512(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p)
#define ALIGNED_MEMCPY(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p) ::FE::aligned_memcpy_with_avx512(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p)

#elif defined(_AVX_)
#define UNALIGNED_MEMSET(dest_ptrc_p, value_p, total_bytes_p) ::FE::unaligned_memset_with_avx(dest_ptrc_p, value_p, total_bytes_p)
#define ALIGNED_MEMSET(dest_ptrc_p, value_p, total_bytes_p) ::FE::aligned_memset_with_avx(dest_ptrc_p, value_p, total_bytes_p)
#define UNALIGNED_MEMCPY(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p) ::FE::unaligned_memcpy_with_avx(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p)
#define ALIGNED_MEMCPY(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p) ::FE::aligned_memcpy_with_avx(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p)

#else
#define UNALIGNED_MEMSET(dest_ptrc_p, value_p, total_bytes_p) ::std::memset(dest_ptrc_p, value_p, total_bytes_p)
#define ALIGNED_MEMSET(dest_ptrc_p, value_p, total_bytes_p) ::std::memset(dest_ptrc_p, value_p, total_bytes_p)
#define UNALIGNED_MEMCPY(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p) ::std::memcpy(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p)
#define ALIGNED_MEMCPY(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p) ::std::memcpy(dest_ptrc_p, source_ptrc_p, bytes_to_copy_p)
#endif


_FORCE_INLINE_ void unaligned_memcpy(void* const dest_memblock_ptrc_p, length_t dest_length_p, size_t dest_element_bytes_p, const void* const source_memblock_ptrc_p, length_t source_length_p, size_t source_element_bytes_p) noexcept
{
	ABORT_IF(dest_memblock_ptrc_p == nullptr, "ERROR: dest_memblock_ptrc_p is nullptr.");
	ABORT_IF(source_memblock_ptrc_p == nullptr, "ERROR: source_memblock_ptrc_p is nullptr.");

	ABORT_IF(dest_length_p == 0, "ERROR: dest_length_p is 0.");
	ABORT_IF(dest_element_bytes_p == 0, "ERROR: dest_bytes_p is 0.");
	ABORT_IF(source_element_bytes_p == 0, "ERROR: source_bytes_p is 0.");

	size_t l_source_size = source_element_bytes_p * source_length_p;
	size_t l_dest_size = dest_element_bytes_p * dest_length_p;

	if (l_source_size >= l_dest_size)
	{
		UNALIGNED_MEMCPY(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_dest_size);
	}
	else
	{
		UNALIGNED_MEMCPY(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_source_size);
	}
}

_FORCE_INLINE_ void aligned_memcpy(void* const dest_memblock_ptrc_p, length_t dest_length_p, size_t dest_element_bytes_p, const void* const source_memblock_ptrc_p, length_t source_length_p, size_t source_element_bytes_p) noexcept
{
	ABORT_IF(dest_memblock_ptrc_p == nullptr, "ERROR: dest_memblock_ptrc_p is nullptr.");
	ABORT_IF(source_memblock_ptrc_p == nullptr, "ERROR: source_memblock_ptrc_p is nullptr.");

	ABORT_IF(dest_length_p == 0, "ERROR: dest_length_p is 0.");
	ABORT_IF(dest_element_bytes_p == 0, "ERROR: dest_bytes_p is 0.");
	ABORT_IF(source_element_bytes_p == 0, "ERROR: source_bytes_p is 0.");

	size_t l_source_size = source_element_bytes_p * source_length_p;
	size_t l_dest_size = dest_element_bytes_p * dest_length_p;

	if (l_source_size >= l_dest_size)
	{
		ALIGNED_MEMCPY(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_dest_size);
	}
	else
	{
		ALIGNED_MEMCPY(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_source_size);
	}
}

_FORCE_INLINE_ void unaligned_memcpy(void* const dest_memblock_ptrc_p, size_t dest_bytes_p, const void* const source_memblock_ptrc_p, size_t source_bytes_p) noexcept
{
	ABORT_IF(dest_memblock_ptrc_p == nullptr, "ERROR: dest_memblock_ptrc_p is nullptr.");
	ABORT_IF(source_memblock_ptrc_p == nullptr, "ERROR: source_memblock_ptrc_p is nullptr.");

	ABORT_IF(dest_bytes_p == 0, "ERROR: dest_bytes_p is 0.");
	ABORT_IF(source_bytes_p == 0, "ERROR: source_bytes_p is 0.");

	if (source_bytes_p >= dest_bytes_p)
	{
		UNALIGNED_MEMCPY(dest_memblock_ptrc_p, source_memblock_ptrc_p, dest_bytes_p);
	}
	else
	{
		UNALIGNED_MEMCPY(dest_memblock_ptrc_p, source_memblock_ptrc_p, source_bytes_p);
	}
}

_FORCE_INLINE_ void aligned_memcpy(void* const dest_memblock_ptrc_p, size_t dest_bytes_p, const void* const source_memblock_ptrc_p, size_t source_bytes_p) noexcept
{
	ABORT_IF(dest_memblock_ptrc_p == nullptr, "ERROR: dest_memblock_ptrc_p is nullptr.");
	ABORT_IF(source_memblock_ptrc_p == nullptr, "ERROR: source_memblock_ptrc_p is nullptr.");

	ABORT_IF(dest_bytes_p == 0, "ERROR: dest_bytes_p is 0.");
	ABORT_IF(source_bytes_p == 0, "ERROR: source_bytes_p is 0.");

	if (source_bytes_p >= dest_bytes_p)
	{
		ALIGNED_MEMCPY(dest_memblock_ptrc_p, source_memblock_ptrc_p, dest_bytes_p);
	}
	else
	{
		ALIGNED_MEMCPY(dest_memblock_ptrc_p, source_memblock_ptrc_p, source_bytes_p);
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
	_ERROR_INVALID_SIZE = 100,
	_ERROR_ILLEGAL_ADDRESS_ALIGNMENT = 101,
	_FATAL_ERROR_NULLPTR = 102,
	_FATAL_ERROR_OUT_OF_RANGE = 103,
	_FATAL_ERROR_OUT_OF_CAPACITY = 104,
	_FATAL_ERROR_ACCESS_VIOLATION = 105,

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
	var::uint64 _length = 0;
};

struct resize_to final
{
	var::uint64 _length = 0;
};

struct extend final
{
	var::uint64 _length = 0;
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
template<uint64 padding_size>
struct align_custom_bytes final
{
	_MAYBE_UNUSED_ static constexpr inline uint16 size = padding_size;
};


#ifdef _AVX512_
template<typename T, class alignment = align_64bytes>
#elif defined(_AVX_)
template<typename T, class alignment = align_32bytes>
#endif
struct alignas(alignment::size) align
{
	T _data;
};


END_NAMESPACE




BEGIN_NAMESPACE(FE)


template<class iterator>
var::boolean memcmp_s(iterator left_iterator_begin_p, iterator left_iterator_end_p, iterator right_iterator_begin_p, iterator right_iterator_end_p) noexcept
{
	static_assert(std::is_class<iterator>::value == true);
	FE_ASSERT(left_iterator_begin_p == nullptr, "ERROR: left_iterator_begin_p is nullptr.");
	FE_ASSERT(left_iterator_end_p == nullptr, "ERROR: left_iterator_end_p is nullptr.");
	FE_ASSERT(right_iterator_begin_p == nullptr, "ERROR: right_iterator_begin_p is nullptr.");
	FE_ASSERT(right_iterator_end_p == nullptr, "ERROR: right_iterator_end_p is nullptr.");

	iterator l_left_iterator_begin = left_iterator_begin_p;

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




template <class T, typename ... arguments>
_DEPRECATED_ _FORCE_INLINE_ void assign(T& dest_object_ref_p, OBJECT_STATUS& dest_bool_mask_ref_p, arguments&& ...arguments_p) noexcept
{
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	if (dest_bool_mask_ref_p == FE::OBJECT_STATUS::_DESTRUCTED)
	{
		new(&dest_object_ref_p) T(arguments_p...);
        dest_bool_mask_ref_p = FE::OBJECT_STATUS::_CONSTRUCTED;
		return;
	}

	dest_object_ref_p = ::std::move(arguments_p...);
}

template <class T>
_DEPRECATED_ _FORCE_INLINE_ void copy_assign(T& dest_object_ref_p, OBJECT_STATUS& dest_bool_mask_ref_p, const T& source_cref_p, _MAYBE_UNUSED_ OBJECT_STATUS source_bool_mask_p) noexcept
{
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(source_bool_mask_p == FE::OBJECT_STATUS::_DESTRUCTED, "ERROR: cannot copy a null object");

	if (dest_bool_mask_ref_p == FE::OBJECT_STATUS::_DESTRUCTED)
	{
		new(&dest_object_ref_p) T(source_cref_p);
		dest_bool_mask_ref_p = FE::OBJECT_STATUS::_CONSTRUCTED;
		return;
	}

	dest_object_ref_p = source_cref_p;
}

template <class T>
_DEPRECATED_ _FORCE_INLINE_ void move_assign(T& dest_object_ref_p, OBJECT_STATUS& dest_bool_mask_ref_p, T&& source_rvalue_reference_p, _MAYBE_UNUSED_ OBJECT_STATUS source_bool_mask_p) noexcept
{
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(source_bool_mask_p == FE::OBJECT_STATUS::_DESTRUCTED, "ERROR: cannot copy a null object");

	if (dest_bool_mask_ref_p == OBJECT_STATUS::_DESTRUCTED)
	{
		new(&dest_object_ref_p) T(std::move(source_rvalue_reference_p));
		dest_bool_mask_ref_p = FE::OBJECT_STATUS::_CONSTRUCTED;
		return;
	}

	dest_object_ref_p = std::move(source_rvalue_reference_p);
}




template <class T>
_DEPRECATED_ _FORCE_INLINE_ void construct(T& dest_ref_p, OBJECT_STATUS& dest_bool_mask_ref_p) noexcept
{
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ref_p == FE::OBJECT_STATUS::_CONSTRUCTED, "ERROR: unable to double-construct.");

	new(&dest_ref_p) T();
	dest_bool_mask_ref_p = FE::OBJECT_STATUS::_CONSTRUCTED;
}

template <class T>
_DEPRECATED_ _FORCE_INLINE_ void copy_construct(T& dest_ref_p, OBJECT_STATUS& dest_bool_mask_ref_p, const T& source_cref_p, _MAYBE_UNUSED_ OBJECT_STATUS source_bool_mask_p) noexcept
{
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ref_p == FE::OBJECT_STATUS::_CONSTRUCTED, "ERROR: unable to double-construct.");
	FE_ASSERT(source_bool_mask_p == FE::OBJECT_STATUS::_DESTRUCTED, "ERROR: unable to copy null object.");

	new(&dest_ref_p) T(source_cref_p);
	dest_bool_mask_ref_p = FE::OBJECT_STATUS::_CONSTRUCTED;
}

template <class T>
_DEPRECATED_ _FORCE_INLINE_ void move_construct(T& dest_ref_p, OBJECT_STATUS& dest_bool_mask_ref_p, T&& source_rvalue_reference_p, _MAYBE_UNUSED_ OBJECT_STATUS source_bool_mask_p) noexcept
{
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ref_p == FE::OBJECT_STATUS::_CONSTRUCTED, "ERROR: unable to double-construct.");
	FE_ASSERT(source_bool_mask_p == FE::OBJECT_STATUS::_DESTRUCTED, "ERROR: unable to copy null object.");

	new(&dest_ref_p) T(source_rvalue_reference_p);
	dest_bool_mask_ref_p = FE::OBJECT_STATUS::_CONSTRUCTED;
}




template <typename iterator, typename ... arguments>
_DEPRECATED_ _FORCE_INLINE_ void assign(iterator begin_p, iterator end_p, OBJECT_STATUS* const boolean_mask_ptrc_p, arguments && ...arguments_p) noexcept
{
	using T = typename iterator::value_type;
	static_assert(std::is_class<iterator>::value == true);
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(boolean_mask_ptrc_p == nullptr, "ERROR: boolean_mask_ptrc_p is nullptr.");
	FE_ASSERT(begin_p == nullptr, "ERROR: begin_p is nullptr.");
	FE_ASSERT(end_p == nullptr, "ERROR: end_p is nullptr.");

	OBJECT_STATUS* l_boolean_mask_ptr = boolean_mask_ptrc_p;

	for (iterator it = begin_p; it != end_p; ++it)
	{
		if (*l_boolean_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
		{
			*it = std::move(arguments_p...);
			++l_boolean_mask_ptr;
		}
		else
		{
			new(it.operator->()) T(arguments_p...);
			*l_boolean_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;
			++l_boolean_mask_ptr;
		}
	}
}

template<class iterator>
_DEPRECATED_ _FORCE_INLINE_ void copy_assign(iterator dest_begin_p, capacity_t dest_length_p, OBJECT_STATUS* const dest_bool_mask_ptrc_p, iterator data_source_begin_p, capacity_t source_data_length_p) noexcept
{
	using T = typename iterator::value_type;
	static_assert(std::is_class<iterator>::value == true);
	static_assert(std::is_copy_assignable<T>::value == true);
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ptrc_p == nullptr, "ERROR: dest_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(data_source_begin_p == nullptr, "ERROR: data_source_begin_p is nullptr.");
	FE_ASSERT(dest_begin_p == nullptr, "ERROR: dest_begin_p is nullptr.");

	OBJECT_STATUS* l_boolean_mask_ptr = dest_bool_mask_ptrc_p;
	iterator l_initializer_list_begin = data_source_begin_p;
	iterator l_dest_iterator_begin = dest_begin_p;

	if (source_data_length_p >= dest_length_p)
	{
		for (var::index_t i = 0; i < dest_length_p; ++i)
		{
			if (*l_boolean_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				*l_dest_iterator_begin = *l_initializer_list_begin;

				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
			else
			{
				new(l_dest_iterator_begin.operator->()) T(*l_initializer_list_begin);
				*l_boolean_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;
				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
		}
		return;
	}
	else
	{
		for (var::index_t i = 0; i < source_data_length_p; ++i)
		{
			if (*l_boolean_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				*l_dest_iterator_begin = *l_initializer_list_begin;

				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
			else
			{
				new(l_dest_iterator_begin.operator->()) T(*l_initializer_list_begin);
				*l_boolean_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;
				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
		}
		return;
	}
}

template<class iterator>
_DEPRECATED_ _FORCE_INLINE_ void move_assign(iterator dest_begin_p, capacity_t dest_length_p, OBJECT_STATUS* const dest_bool_mask_ptrc_p, iterator data_source_begin_p, capacity_t source_data_length_p) noexcept
{
	using T = typename iterator::value_type;
	static_assert(std::is_class<iterator>::value == true);
	static_assert(std::is_move_assignable<T>::value == true);
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ptrc_p == nullptr, "ERROR: dest_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(data_source_begin_p == nullptr, "ERROR: data_source_begin_p is nullptr.");
	FE_ASSERT(dest_begin_p == nullptr, "ERROR: dest_begin_p is nullptr.");

	OBJECT_STATUS* l_boolean_mask_ptr = dest_bool_mask_ptrc_p;
	iterator l_initializer_list_begin = data_source_begin_p;
	iterator l_dest_iterator_begin = dest_begin_p;

	if (source_data_length_p >= dest_length_p)
	{
		for (var::index_t i = 0; i < dest_length_p; ++i)
		{
			if (*l_boolean_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				*l_dest_iterator_begin = std::move(*l_initializer_list_begin);

				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
			else
			{
				new(l_dest_iterator_begin.operator->()) T(std::move(*l_initializer_list_begin));
				*l_boolean_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;
				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
		}
		return;
	}
	else
	{
		for (var::index_t i = 0; i < source_data_length_p; ++i)
		{
			if (*l_boolean_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				*l_dest_iterator_begin = std::move(*l_initializer_list_begin);

				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
			else
			{
				new(l_dest_iterator_begin.operator->()) T(std::move(*l_initializer_list_begin));
				*l_boolean_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;
				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
		}
		return;
	}
}

template<class iterator>
_DEPRECATED_ _FORCE_INLINE_ void copy_assign(iterator dest_begin_p, capacity_t dest_length_p, OBJECT_STATUS* const dest_bool_mask_ptrc_p, iterator source_data_begin_p, capacity_t source_data_length_p, const OBJECT_STATUS* const source_data_bool_mask_ptrc_p) noexcept
{
	using T = typename iterator::value_type;
	static_assert(std::is_class<iterator>::value == true);
	static_assert(std::is_copy_assignable<T>::value == true);
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ptrc_p == nullptr, "ERROR: dest_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(source_data_begin_p == nullptr, "ERROR: source_data_begin_p is nullptr.");
	FE_ASSERT(source_data_bool_mask_ptrc_p == nullptr, "ERROR: source_data_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(dest_begin_p == nullptr, "ERROR: dest_begin_p is nullptr.");

	OBJECT_STATUS* l_dest_bool_mask_ptr = dest_bool_mask_ptrc_p;
	const OBJECT_STATUS* l_source_bool_mask_ptr = source_data_bool_mask_ptrc_p;

	iterator l_initializer_list_begin = source_data_begin_p;
	iterator l_dest_iterator_begin = dest_begin_p;

	if (source_data_length_p >= dest_length_p)
	{
		for (var::index_t i = 0; i < dest_length_p; ++i)
		{
			if (*l_dest_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED && *l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				*l_dest_iterator_begin = *l_initializer_list_begin;

				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
			else if (*l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(*l_initializer_list_begin);
				*l_dest_bool_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
		}
		return;
	}
	else
	{
		for (var::index_t i = 0; i < source_data_length_p; ++i)
		{
			if (*l_dest_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED && *l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				*l_dest_iterator_begin = *l_initializer_list_begin;

				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
			else if (*l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(*l_initializer_list_begin);
				*l_dest_bool_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
		}
		return;
	}
}

template<class iterator>
_DEPRECATED_ _FORCE_INLINE_ void move_assign(iterator dest_begin_p, capacity_t dest_length_p, OBJECT_STATUS* const dest_bool_mask_ptrc_p, iterator source_data_begin_p, capacity_t source_data_length_p, OBJECT_STATUS* const source_data_bool_mask_ptrc_p) noexcept
{
	using T = typename iterator::value_type;
	static_assert(std::is_class<iterator>::value == true);
	static_assert(std::is_move_assignable<T>::value == true);
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ptrc_p == nullptr, "ERROR: dest_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(source_data_begin_p == nullptr, "ERROR: source_data_begin_p is nullptr.");
	FE_ASSERT(source_data_bool_mask_ptrc_p == nullptr, "ERROR: source_data_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(dest_begin_p == nullptr, "ERROR: dest_begin_p is nullptr.");

	OBJECT_STATUS* l_dest_bool_mask_ptr = dest_bool_mask_ptrc_p;
	OBJECT_STATUS* l_source_bool_mask_ptr = source_data_bool_mask_ptrc_p;

	iterator l_initializer_list_begin = source_data_begin_p;
	iterator l_dest_iterator_begin = dest_begin_p;

	if (source_data_length_p >= dest_length_p)
	{
		for (var::index_t i = 0; i < dest_length_p; ++i)
		{
			if (*l_dest_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED && *l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				*l_dest_iterator_begin = std::move(*l_initializer_list_begin);

				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
			else if(*l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(std::move(*l_initializer_list_begin));
				*l_dest_bool_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
		}
		return;
	}
	else
	{
		for (var::index_t i = 0; i < source_data_length_p; ++i)
		{
			if (*l_dest_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED && *l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				*l_dest_iterator_begin = std::move(*l_initializer_list_begin);

				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
			else if(*l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(std::move(*l_initializer_list_begin));
				*l_dest_bool_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
		}
		return;
	}
}




template <class T>
_DEPRECATED_ _FORCE_INLINE_ void destruct(T& dest_ref_p, OBJECT_STATUS& dest_bool_mask_ref_p) noexcept
{
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ref_p == FE::OBJECT_STATUS::_DESTRUCTED, "ERROR: unable to destruct.");

	dest_ref_p.~T();
	dest_bool_mask_ref_p = FE::OBJECT_STATUS::_DESTRUCTED;
}

template<class iterator>
_DEPRECATED_ _FORCE_INLINE_ void destruct(iterator begin_p, iterator end_p, OBJECT_STATUS* const boolean_mask_ptrc_p) noexcept
{
	using T = typename iterator::value_type;
	static_assert(std::is_class<iterator>::value == true);
	static_assert(std::is_destructible<T>::value == true);
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(boolean_mask_ptrc_p == nullptr, "ERROR: boolean_mask_ptrc_p is nullptr.");
	FE_ASSERT(begin_p == nullptr, "ERROR: begin_p is nullptr.");
	FE_ASSERT(end_p == nullptr, "ERROR: end_p is nullptr.");

	OBJECT_STATUS* l_boolean_mask_ptr = boolean_mask_ptrc_p;

	for (iterator it = begin_p; it != end_p; ++it)
	{
		if (*l_boolean_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
		{
			it->~T();
			*l_boolean_mask_ptr = FE::OBJECT_STATUS::_DESTRUCTED;
			++l_boolean_mask_ptr;
		}
	}
}




template<class iterator>
_DEPRECATED_ _FORCE_INLINE_ void copy_construct(iterator dest_begin_p, capacity_t dest_length_p, OBJECT_STATUS* const dest_bool_mask_ptrc_p, iterator data_source_begin_p, capacity_t source_data_length_p) noexcept
{
	using T = typename iterator::value_type;
	static_assert(std::is_class<iterator>::value == true);
	static_assert(std::is_copy_constructible<T>::value == true);
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ptrc_p == nullptr, "ERROR: dest_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(data_source_begin_p == nullptr, "ERROR: data_source_begin_p is nullptr.");
	FE_ASSERT(dest_begin_p == nullptr, "ERROR: dest_begin_p is nullptr.");

	OBJECT_STATUS* l_boolean_mask_ptr = dest_bool_mask_ptrc_p;
	iterator l_initializer_list_begin = data_source_begin_p;
	iterator l_dest_iterator_begin = dest_begin_p;

	if (source_data_length_p >= dest_length_p)
	{
		for (var::index_t i = 0; i < dest_length_p; ++i)
		{
			if (*l_boolean_mask_ptr == FE::OBJECT_STATUS::_DESTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(*l_initializer_list_begin);
				*l_boolean_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;

				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
		}
		return;
	}
	else
	{
		for (var::index_t i = 0; i < source_data_length_p; ++i)
		{
			if (*l_boolean_mask_ptr == FE::OBJECT_STATUS::_DESTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(*l_initializer_list_begin);
				*l_boolean_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;

				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
		}
		return;
	}
}

template<class iterator>
_DEPRECATED_ _FORCE_INLINE_ void move_construct(iterator dest_begin_p, capacity_t dest_length_p, OBJECT_STATUS* const dest_bool_mask_ptrc_p, iterator data_source_begin_p, capacity_t source_data_length_p) noexcept
{
	using T = typename iterator::value_type;
	static_assert(std::is_class<iterator>::value == true);
	static_assert(std::is_move_constructible<T>::value == true);
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ptrc_p == nullptr, "ERROR: dest_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(data_source_begin_p == nullptr, "ERROR: data_source_begin_p is nullptr.");
	FE_ASSERT(dest_begin_p == nullptr, "ERROR: dest_begin_p is nullptr.");

	OBJECT_STATUS* l_boolean_mask_ptr = dest_bool_mask_ptrc_p;
	iterator l_initializer_list_begin = data_source_begin_p;
	iterator l_dest_iterator_begin = dest_begin_p;

	if (source_data_length_p >= dest_length_p)
	{
		for (var::index_t i = 0; i < dest_length_p; ++i)
		{
			if (*l_boolean_mask_ptr == FE::OBJECT_STATUS::_DESTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(std::move(*l_initializer_list_begin));
				*l_boolean_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;

				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
		}
		return;
	}
	else
	{
		for (var::index_t i = 0; i < source_data_length_p; ++i)
		{
			if (*l_boolean_mask_ptr == FE::OBJECT_STATUS::_DESTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(std::move(*l_initializer_list_begin));
				*l_boolean_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;

				++l_boolean_mask_ptr;
				++l_initializer_list_begin;
				++l_dest_iterator_begin;
			}
		}
		return;
	}
}

template<class iterator>
_DEPRECATED_ _FORCE_INLINE_ void copy_construct(iterator dest_begin_p, capacity_t dest_length_p, OBJECT_STATUS* const dest_bool_mask_ptrc_p, iterator source_data_begin_p, capacity_t source_data_length_p, const OBJECT_STATUS* const source_data_bool_mask_ptrc_p) noexcept
{
	using T = typename iterator::value_type;
	static_assert(std::is_class<iterator>::value == true);
	static_assert(std::is_copy_constructible<T>::value == true);
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ptrc_p == nullptr, "ERROR: dest_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(source_data_begin_p == nullptr, "ERROR: source_data_begin_p is nullptr.");
	FE_ASSERT(source_data_bool_mask_ptrc_p == nullptr, "ERROR: source_data_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(dest_begin_p == nullptr, "ERROR: dest_begin_p is nullptr.");

	OBJECT_STATUS* l_dest_bool_mask_ptr = dest_bool_mask_ptrc_p;
	const OBJECT_STATUS* l_source_bool_mask_ptr = source_data_bool_mask_ptrc_p;

	iterator l_initializer_list_begin = source_data_begin_p;
	iterator l_dest_iterator_begin = dest_begin_p;

	if (source_data_length_p >= dest_length_p)
	{
		for (var::index_t i = 0; i < dest_length_p; ++i)
		{
			if (*l_dest_bool_mask_ptr == FE::OBJECT_STATUS::_DESTRUCTED && *l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(*l_initializer_list_begin);
				*l_dest_bool_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;

				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
		}
		return;
	}
	else
	{
		for (var::index_t i = 0; i < source_data_length_p; ++i)
		{
			if (*l_dest_bool_mask_ptr == FE::OBJECT_STATUS::_DESTRUCTED && *l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(*l_initializer_list_begin);
				*l_dest_bool_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;

				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
		}
		return;
	}
}

template<class iterator>
_DEPRECATED_ _FORCE_INLINE_ void move_construct(iterator dest_begin_p, capacity_t dest_length_p, OBJECT_STATUS* const dest_bool_mask_ptrc_p, iterator source_data_begin_p, capacity_t source_data_length_p, OBJECT_STATUS* const source_data_bool_mask_ptrc_p) noexcept
{
	using T = typename iterator::value_type;
	static_assert(std::is_class<iterator>::value == true);
	static_assert(std::is_move_constructible<T>::value == true);
	static_assert(FE::is_trivially_constructible_and_destructible<T>::value == FE::TYPE_TRIVIALITY::_NOT_TRIVIAL, "WARNING: T must not be trivially constructible and destructible. This function call has no effect and is a waste of computing resource");
	FE_ASSERT(dest_bool_mask_ptrc_p == nullptr, "ERROR: dest_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(source_data_begin_p == nullptr, "ERROR: source_data_begin_p is nullptr.");
	FE_ASSERT(source_data_bool_mask_ptrc_p == nullptr, "ERROR: source_data_bool_mask_ptrc_p is nullptr.");
	FE_ASSERT(dest_begin_p == nullptr, "ERROR: dest_begin_p is nullptr.");

	OBJECT_STATUS* l_dest_bool_mask_ptr = dest_bool_mask_ptrc_p;
	OBJECT_STATUS* l_source_bool_mask_ptr = source_data_bool_mask_ptrc_p;

	iterator l_initializer_list_begin = source_data_begin_p;
	iterator l_dest_iterator_begin = dest_begin_p;

	if (source_data_length_p >= dest_length_p)
	{
		for (var::index_t i = 0; i < dest_length_p; ++i)
		{
			if (*l_dest_bool_mask_ptr == FE::OBJECT_STATUS::_DESTRUCTED && *l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(std::move(*l_initializer_list_begin));
				*l_dest_bool_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;

				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
		}
		return;
	}
	else
	{
		for (var::index_t i = 0; i < source_data_length_p; ++i)
		{
			if (*l_dest_bool_mask_ptr == FE::OBJECT_STATUS::_DESTRUCTED && *l_source_bool_mask_ptr == FE::OBJECT_STATUS::_CONSTRUCTED)
			{
				new(l_dest_iterator_begin.operator->()) T(std::move(*l_initializer_list_begin));
				*l_dest_bool_mask_ptr = FE::OBJECT_STATUS::_CONSTRUCTED;

				++l_initializer_list_begin;
				++l_dest_iterator_begin;

				++l_dest_bool_mask_ptr;
				++l_source_bool_mask_ptr;
			}
		}
		return;
	}
}


END_NAMESPACE








BEGIN_NAMESPACE(FE)

// refactor smart_ptrs and add new features to type_trait
// T must be (copy/move) (constructible/assignable). 
// 
// void construct(T* const ptrc_p, T value_p) noecept;
// void construct(T* const dest_begin_ptrc_p, T* const dest_end_ptrc_p, T value_p) noecept;
// void construct(T* const dest_begin_ptrc_p, T* const dest_end_ptrc_p, T* const begin_source_p, T* const end_source_p) noecept;
// 
// void destruct(T* const ptrc_p) noecept;
// void destruct(T* const begin_ptrc_p, T* const end_ptrc_p) noecept;

template<typename T, TYPE_TRIVIALITY is_trivial = FE::is_trivially_constructible_and_destructible<T>::value>
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




template<typename T, class allocated_from, TYPE_TRIVIALITY is_trivial = allocated_from::is_trivially_constructible_and_destructible>
class type_trait;

template<typename T, class allocated_from>
class type_trait<T, allocated_from, TYPE_TRIVIALITY::_TRIVIAL>
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
		if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == true)
		{
			FE::aligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
		else if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == false)
		{
			FE::unaligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
	}
	
	_FORCE_INLINE_ static void copy_construct(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == true)
		{
			ALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
		else if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == false)
		{
			UNALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
	}


	_FORCE_INLINE_ static void move_construct(T* dest_ptr_p, count_t dest_count_p, T* source_ptr_p, count_t source_count_p) noexcept
	{
		if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == true)
		{
			FE::aligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
		else if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == false)
		{
			FE::unaligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
	}

	_FORCE_INLINE_ static void move_construct(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == true)
		{
			ALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
		else if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == false)
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
		if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == true)
		{
			FE::aligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
		else if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == false)
		{
			FE::unaligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
	}

	_FORCE_INLINE_ static void copy_assign(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == true)
		{
			ALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
		else if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == false)
		{
			UNALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
	}


	_FORCE_INLINE_ static void move_assign(T* dest_ptr_p, count_t dest_count_p, T* source_ptr_p, count_t source_count_p) noexcept
	{
		if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == true)
		{
			FE::aligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
		else if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == false)
		{
			FE::unaligned_memcpy(dest_ptr_p, sizeof(T) * dest_count_p, source_ptr_p, sizeof(T) * source_count_p);
		}
	}

	_FORCE_INLINE_ static void move_assign(T* dest_ptr_p, T* source_ptr_p, count_t count_to_copy_or_move_p) noexcept
	{
		if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == true)
		{
			ALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
		else if constexpr (allocated_from::is_allocated_from_an_address_aligned_allocator == false)
		{
			UNALIGNED_MEMCPY(dest_ptr_p, source_ptr_p, sizeof(T) * count_to_copy_or_move_p);
		}
	}
};


template<typename T, class allocated_from>
class type_trait<T, allocated_from, TYPE_TRIVIALITY::_NOT_TRIVIAL>
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

		FE_ASSERT(dest_count_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(dest_count_p));
		FE_ASSERT(source_count_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(source_count_p));

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

		FE_ASSERT(count_to_copy_or_move_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(count_to_copy_or_move_p));

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

		FE_ASSERT(dest_count_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(dest_count_p));
		FE_ASSERT(source_count_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(source_count_p));

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

		FE_ASSERT(count_to_copy_or_move_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(count_to_copy_or_move_p));

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

		FE_ASSERT(dest_count_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(dest_count_p));
		FE_ASSERT(source_count_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(source_count_p));

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

		FE_ASSERT(count_to_copy_or_move_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(count_to_copy_or_move_p));

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

		FE_ASSERT(dest_count_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(dest_count_p));
		FE_ASSERT(source_count_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(source_count_p));

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

		FE_ASSERT(count_to_copy_or_move_p == 0, "MEMORY_ERROR_1XX::_ERROR_INVALID_SIZE: ${%s@0} is zero", TO_STRING(count_to_copy_or_move_p));

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

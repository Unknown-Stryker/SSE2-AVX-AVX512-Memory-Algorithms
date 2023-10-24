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

#include <FE/core/prerequisites.h>
#include <FE/core/algorithm/math.h>
#include <FE/core/iterator.hxx>
#include <immintrin.h>




BEGIN_NAMESPACE(FE)


enum struct OBJECT_STATUS : boolean
{
	_CONSTRUCTED = true,
	_DESTRUCTED = false
};

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


struct reserve final
{
	size_t _value = 0;
};

struct resize_to final
{
	size_t _value = 0;
};

struct extend final
{
	size_t _value = 0;
};

struct count final
{
	size_t _value = 0;
};


struct align_4bytes final
{
	_MAYBE_UNUSED_ static constexpr size_t size = 4;
};

struct align_8bytes final
{
	_MAYBE_UNUSED_ static constexpr size_t size = 8;
};

struct align_16bytes final
{
	_MAYBE_UNUSED_ static constexpr size_t size = 16;
};

struct align_32bytes final
{
	_MAYBE_UNUSED_ static constexpr size_t size = 32;
};

struct align_64bytes final
{
	_MAYBE_UNUSED_ static constexpr size_t size = 64;
};

struct align_128bytes final
{
	_MAYBE_UNUSED_ static constexpr size_t size = 128;
};

struct align_CPU_L1_cache_line final
{
	_MAYBE_UNUSED_ static constexpr size_t size = std::hardware_destructive_interference_size;
};


// it contains memory padding size.
template<uint64 PaddingSize>
struct align_custom_bytes final
{
	_MAYBE_UNUSED_ static constexpr inline uint16 size = PaddingSize;
};


struct SIMD_auto_alignment
{
#ifdef _AVX512_
	using alignment_type = align_64bytes;
#elif defined(_AVX_)
	using alignment_type = align_32bytes;
#else
	using alignment_type = align_16bytes;
#endif
};


template<typename T, class Alignment = typename FE::SIMD_auto_alignment::alignment_type>
struct alignas(Alignment::size) aligned
{
	T _data;
};


enum struct ADDRESS : boolean
{
	_NOT_ALIGNED = false,
	_ALIGNED = true
};


#ifdef _AVX512_
_FORCE_INLINE_ void unaligned_memset_with_avx512(void* const out_dest_pointer_p, int8 value_p, size_t total_bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(total_bytes_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(total_bytes_p));

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(out_dest_pointer_p);
	const __m512i l_m512i_value_to_be_assigned = _mm512_set1_epi8(value_p);

	var::size_t l_leftover_bytes = MODULO_BY_64(total_bytes_p);
	size_t l_avx512_operation_count = MODULO_BY_64(total_bytes_p);

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
	size_t l_avx512_operation_count = MODULO_BY_64(total_bytes_p);

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
	FE_ASSERT(out_dest_pointer_p == source_pointer_p, "Assertion Failure: A destination pointer and a source pointer cannot point to the same address.");

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(out_dest_pointer_p);
	const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_64(bytes_to_copy_p);
	size_t l_avx512_operation_count = MODULO_BY_64(bytes_to_copy_p);

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
	FE_ASSERT(out_dest_pointer_p == source_pointer_p, "Assertion Failure: A destination pointer and a source pointer cannot point to the same address.");

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(out_dest_pointer_p);
	const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_64(bytes_to_copy_p);
	size_t l_avx512_operation_count = MODULO_BY_64(bytes_to_copy_p);

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
	FE_ASSERT(out_dest_pointer_p == source_pointer_p, "Assertion Failure: A destination pointer and a source pointer cannot point to the same address.");

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(out_dest_pointer_p);
	const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_64(bytes_to_copy_p);
	size_t l_avx512_operation_count = MODULO_BY_64(bytes_to_copy_p);

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
	FE_ASSERT(out_dest_pointer_p == source_pointer_p, "Assertion Failure: A destination pointer and a source pointer cannot point to the same address.");

	__m512i* l_m512i_dest_ptr = static_cast<__m512i*>(out_dest_pointer_p);
	const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_64(bytes_to_copy_p);
	size_t l_avx512_operation_count = MODULO_BY_64(bytes_to_copy_p);

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
	// _mm256_storeu_si256()
	size_t l_avx_operation_count = DIVIDE_BY_32(total_bytes_p);

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
	// _mm256_store_si256()
	size_t l_avx_operation_count = DIVIDE_BY_32(total_bytes_p);

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
	FE_ASSERT(out_dest_pointer_p == source_pointer_p, "Assertion Failure: A destination pointer and a source pointer cannot point to the same address.");

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(out_dest_pointer_p);
	const __m256i* l_m256i_source_ptr = static_cast<const __m256i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
	// _mm256_storeu_si256() and _mm256_loadu_si256()
	size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p);

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
	FE_ASSERT(out_dest_pointer_p == source_pointer_p, "Assertion Failure: A destination pointer and a source pointer cannot point to the same address.");

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(out_dest_pointer_p);
	const __m256i* l_m256i_source_ptr = static_cast<const __m256i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
	// _mm256_store_si256() and _mm256_load_si256()
	size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p);

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
	FE_ASSERT(out_dest_pointer_p == source_pointer_p, "Assertion Failure: A destination pointer and a source pointer cannot point to the same address.");

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(out_dest_pointer_p);
	const __m256i* l_m256i_source_ptr = static_cast<const __m256i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
	// _mm256_store_si256() and _mm256_loadu_si256()
	size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p);

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
	FE_ASSERT(out_dest_pointer_p == source_pointer_p, "Assertion Failure: A destination pointer and a source pointer cannot point to the same address.");

	__m256i* l_m256i_dest_ptr = static_cast<__m256i*>(out_dest_pointer_p);
	const __m256i* l_m256i_source_ptr = static_cast<const __m256i*>(source_pointer_p);

	var::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
	// _mm256_storeu_si256() and _mm256_load_si256()
	size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p);

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


_FORCE_INLINE_ void unaligned_memmove_with_avx(void* const out_dest_pointer_p, const void* const source_pointer_p, FE::size_t bytes_to_move_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_pointer_p));
	FE_ASSERT(bytes_to_move_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(bytes_to_move_p));
	FE_ASSERT(out_dest_pointer_p == source_pointer_p, "Assertion Failure: A destination pointer and a source pointer cannot point to the same address.");

	if (source_pointer_p < out_dest_pointer_p)
	{
		var::byte* l_dest_byte_ptr = static_cast<var::byte*>(out_dest_pointer_p) + (bytes_to_move_p - 1);
		byte* l_source_byte_ptr = static_cast<byte*>(source_pointer_p) + (bytes_to_move_p - 1);

		{
			size_t l_leftover_bytes_to_copy_by_byte = MODULO_BY_16(bytes_to_move_p);

			for (var::size_t i = 0; i != l_leftover_bytes_to_copy_by_byte; ++i)
			{
				*l_dest_byte_ptr = *l_source_byte_ptr;
				--l_dest_byte_ptr;
				--l_source_byte_ptr;
			}
		}

		var::size_t l_avx_16bytes_operation_count = DIVIDE_BY_16(bytes_to_move_p);

		__m256i* l_m256i_dest_ptr = reinterpret_cast<__m256i*>(l_dest_byte_ptr + 1) - 1;
		const __m256i* l_m256i_source_ptr = reinterpret_cast<const __m256i*>(l_source_byte_ptr + 1) - 1;
	
		for (; l_avx_16bytes_operation_count > 1; l_avx_16bytes_operation_count -= 2)
		{
			_mm256_storeu_si256(l_m256i_dest_ptr, _mm256_loadu_si256(l_m256i_source_ptr));
			--l_m256i_dest_ptr;
			--l_m256i_source_ptr;
		}

		__m128i* l_m128i_dest_ptr = reinterpret_cast<__m128i*>(l_m256i_dest_ptr) + 1;
		const __m128i* l_m128i_source_ptr = reinterpret_cast<const __m128i*>(l_m256i_source_ptr) + 1;

		for (; l_avx_16bytes_operation_count > 0; --l_avx_16bytes_operation_count)
		{
			_mm_storeu_si128(l_m128i_dest_ptr, _mm_loadu_si128(l_m128i_source_ptr));
			--l_m128i_dest_ptr;
			--l_m128i_source_ptr;
		}
	}
	else
	{
		unaligned_memcpy_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_move_p);
	}
}

_FORCE_INLINE_ void aligned_memmove_with_avx(void* const out_dest_pointer_p, const void* const source_pointer_p, FE::size_t bytes_to_move_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(source_pointer_p));
	FE_ASSERT(bytes_to_move_p == 0, "${%s@0}: ${%s@1} is 0.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_INVALID_SIZE), TO_STRING(bytes_to_move_p));
	FE_ASSERT(out_dest_pointer_p == source_pointer_p, "Assertion Failure: A destination pointer and a source pointer cannot point to the same address.");
	FE_ASSERT(MODULO_BY_32(reinterpret_cast<uintptr_t>(out_dest_pointer_p)) != 0, "${%s@0}: ${%s@1} is not aligned by 32.", TO_STRING(MEMORY_ERROR_1XX::_ERROR_ILLEGAL_ADDRESS_ALIGNMENT), TO_STRING(out_dest_pointer_p));

	if (source_pointer_p < out_dest_pointer_p)
	{
		var::byte* l_dest_byte_ptr = static_cast<var::byte*>(out_dest_pointer_p) + (bytes_to_move_p - 1);
		byte* l_source_byte_ptr = static_cast<byte*>(source_pointer_p) + (bytes_to_move_p - 1);

		{
			size_t l_leftover_bytes_to_copy_by_byte = MODULO_BY_16(bytes_to_move_p);

			for (var::size_t i = 0; i != l_leftover_bytes_to_copy_by_byte; ++i)
			{
				*l_dest_byte_ptr = *l_source_byte_ptr;
				--l_dest_byte_ptr;
				--l_source_byte_ptr;
			}
}

		var::size_t l_avx_16bytes_operation_count = DIVIDE_BY_16(bytes_to_move_p);

		__m256i* l_m256i_dest_ptr = reinterpret_cast<__m256i*>(l_dest_byte_ptr + 1) - 1;
		const __m256i* l_m256i_source_ptr = reinterpret_cast<const __m256i*>(l_source_byte_ptr + 1) - 1;

		for (; l_avx_16bytes_operation_count > 1; l_avx_16bytes_operation_count -= 2)
		{
			_mm256_store_si256(l_m256i_dest_ptr, _mm256_loadu_si256(l_m256i_source_ptr));
			--l_m256i_dest_ptr;
			--l_m256i_source_ptr;
		}

		__m128i* l_m128i_dest_ptr = reinterpret_cast<__m128i*>(l_m256i_dest_ptr) + 1;
		const __m128i* l_m128i_source_ptr = reinterpret_cast<const __m128i*>(l_m256i_source_ptr) + 1;

		for (; l_avx_16bytes_operation_count > 0; --l_avx_16bytes_operation_count)
		{
			_mm_store_si128(l_m128i_dest_ptr, _mm_loadu_si128(l_m128i_source_ptr));
			--l_m128i_dest_ptr;
			--l_m128i_source_ptr;
		}
	}
	else
	{
		aligned_memcpy_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_move_p);
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
#define UNALIGNED_MEMMOVE(out_dest_pointer_p, source_pointer_p, bytes_to_move_p) /*::FE::unaligned_memmove_with_avx512(out_dest_pointer_p, source_pointer_p, bytes_to_move_p)*/
#define ALIGNED_MEMMOVE(out_dest_pointer_p, source_pointer_p, bytes_to_move_p) /*::FE::aligned_memmove_with_avx512(out_dest_pointer_p, source_pointer_p, bytes_to_move_p)*/

#elif defined(_AVX_)
#define UNALIGNED_MEMSET(out_dest_pointer_p, value_p, total_bytes_p) ::FE::unaligned_memset_with_avx(out_dest_pointer_p, value_p, total_bytes_p)
#define ALIGNED_MEMSET(out_dest_pointer_p, value_p, total_bytes_p) ::FE::aligned_memset_with_avx(out_dest_pointer_p, value_p, total_bytes_p)
#define UNALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::unaligned_memcpy_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::aligned_memcpy_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define DEST_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::dest_aligned_memcpy_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define SOURCE_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::FE::source_aligned_memcpy_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define UNALIGNED_MEMMOVE(out_dest_pointer_p, source_pointer_p, bytes_to_move_p) ::FE::unaligned_memmove_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_move_p)
#define ALIGNED_MEMMOVE(out_dest_pointer_p, source_pointer_p, bytes_to_move_p) ::FE::aligned_memmove_with_avx(out_dest_pointer_p, source_pointer_p, bytes_to_move_p)

#else
#define UNALIGNED_MEMSET(out_dest_pointer_p, value_p, total_bytes_p) ::std::memset(out_dest_pointer_p, value_p, total_bytes_p)
#define ALIGNED_MEMSET(out_dest_pointer_p, value_p, total_bytes_p) ::std::memset(out_dest_pointer_p, value_p, total_bytes_p)
#define UNALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::std::memcpy(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::std::memcpy(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define DEST_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::std::memcpy(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define SOURCE_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p) ::std::memcpy(out_dest_pointer_p, source_pointer_p, bytes_to_copy_p)
#define UNALIGNED_MEMMOVE(out_dest_pointer_p, source_pointer_p, bytes_to_move_p) ::std::memmove(out_dest_pointer_p, source_pointer_p, bytes_to_move_p)
#define ALIGNED_MEMMOVE(out_dest_pointer_p, source_pointer_p, bytes_to_move_p) ::std::memmove(out_dest_pointer_p, source_pointer_p, bytes_to_move_p)
#endif




template<class ConstIterator>
_FORCE_INLINE_ boolean memcmp(ConstIterator left_iterator_begin_p, ConstIterator left_iterator_end_p, ConstIterator right_iterator_begin_p, ConstIterator right_iterator_end_p) noexcept
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
_FORCE_INLINE_ void memcpy(void* out_dest_pointer_p, size_t dest_capacity_in_bytes_p, const void* source_pointer_p, count_t bytes_p) noexcept
	{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
	FE_ASSERT(dest_capacity_in_bytes_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(dest_capacity_in_bytes_p));
	FE_ASSERT(bytes_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(bytes_p));

	if constexpr (DestAddressAlignment == ADDRESS::_ALIGNED && SourceAddressAlignment == ADDRESS::_ALIGNED)
	{
		ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, MIN(dest_capacity_in_bytes_p, bytes_p));
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_ALIGNED && SourceAddressAlignment == ADDRESS::_NOT_ALIGNED)
	{
		DEST_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, MIN(dest_capacity_in_bytes_p, bytes_p));
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_NOT_ALIGNED && SourceAddressAlignment == ADDRESS::_ALIGNED)
	{
		SOURCE_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, MIN(dest_capacity_in_bytes_p, bytes_p));
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_NOT_ALIGNED && SourceAddressAlignment == ADDRESS::_NOT_ALIGNED)
	{
		UNALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, MIN(dest_capacity_in_bytes_p, bytes_p));
	}
}

template<ADDRESS DestAddressAlignment = ADDRESS::_NOT_ALIGNED, ADDRESS SourceAddressAlignment = ADDRESS::_NOT_ALIGNED>
_FORCE_INLINE_ void memcpy(void* out_dest_pointer_p, const void* source_pointer_p, count_t bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
	FE_ASSERT(bytes_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(bytes_p));

	if constexpr (DestAddressAlignment == ADDRESS::_ALIGNED && SourceAddressAlignment == ADDRESS::_ALIGNED)
	{
		ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_p);
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_ALIGNED && SourceAddressAlignment == ADDRESS::_NOT_ALIGNED)
	{
		DEST_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_p);
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_NOT_ALIGNED && SourceAddressAlignment == ADDRESS::_ALIGNED)
	{
		SOURCE_ALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_p);
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_NOT_ALIGNED && SourceAddressAlignment == ADDRESS::_NOT_ALIGNED)
	{
		UNALIGNED_MEMCPY(out_dest_pointer_p, source_pointer_p, bytes_p);
	}
}

template<ADDRESS DestAddressAlignment = ADDRESS::_NOT_ALIGNED>
_FORCE_INLINE_ void memset(void* out_dest_pointer_p, int8 value_p, count_t bytes_p) noexcept
{
	FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
	FE_ASSERT(bytes_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(bytes_p));

	if constexpr (DestAddressAlignment == ADDRESS::_ALIGNED)
	{
		ALIGNED_MEMSET(out_dest_pointer_p, (int8)value_p, bytes_p);
	}
	else if constexpr (DestAddressAlignment == ADDRESS::_NOT_ALIGNED)
	{
		UNALIGNED_MEMSET(out_dest_pointer_p, (int8)value_p, bytes_p);
	}
}




template<typename T, ADDRESS DestAddressAlignment = ADDRESS::_NOT_ALIGNED, ADDRESS SourceAddressAlignment = ADDRESS::_NOT_ALIGNED, TYPE_TRIVIALITY IsTrivial = FE::is_trivial<T>::value>
class memory_traits;


template<typename T, ADDRESS DestAddressAlignment, ADDRESS SourceAddressAlignment>
class memory_traits<T, DestAddressAlignment, SourceAddressAlignment, TYPE_TRIVIALITY::_TRIVIAL> final
{
public:
	_MAYBE_UNUSED_ static constexpr inline TYPE_TRIVIALITY is_trivial = TYPE_TRIVIALITY::_TRIVIAL;


	_FORCE_INLINE_ static void construct(_MAYBE_UNUSED_ T& out_dest_p) noexcept
	{
	}

	_FORCE_INLINE_ static void construct(T& out_dest_p, T value_p) noexcept
	{
		out_dest_p = std::move(value_p);
	}

	template<class Iterator>
	_FORCE_INLINE_ static void construct(Iterator in_out_dest_first_p, Iterator in_out_dest_last_p, const T& value_p) noexcept
	{
		static_assert(std::is_constructible<T>::value == true, "static assertion failed: The typename T must be copy constructible.");

		FE_ASSERT(in_out_dest_first_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(in_out_dest_last_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));

		FE_ASSERT(in_out_dest_first_p > in_out_dest_last_p, "${%s@0}: The begin iterator ${%s@1} must be pointing at the first element of a container.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_ILLEGAL_POSITION), TO_STRING(in_out_dest_first_p));
		
		if constexpr (std::is_class<Iterator>::value == true)
		{
			if constexpr (std::is_same<typename Iterator::iterator_category, typename FE::contiguous_iterator<typename Iterator::value_type>::category>::value == true && sizeof(T) == sizeof(std::byte))
			{
				UNALIGNED_MEMSET(iterator_cast<T*>(in_out_dest_first_p), (int8)value_p, (in_out_dest_last_p - in_out_dest_first_p) * sizeof(T));
			}
			else
			{
				while (in_out_dest_first_p != in_out_dest_last_p)
				{
					*in_out_dest_first_p = value_p;
					++in_out_dest_first_p;
				}
			}
		}
		else if constexpr ((std::is_pointer<Iterator>::value == true) && (sizeof(T) == sizeof(std::byte)))
		{
			UNALIGNED_MEMSET(iterator_cast<T*>(in_out_dest_first_p), (int8)value_p, (in_out_dest_last_p - in_out_dest_first_p) * sizeof(T));
		}
		else
		{
			while (in_out_dest_first_p != in_out_dest_last_p)
			{
				*in_out_dest_first_p = value_p;
				++in_out_dest_first_p;
			}
		}
	}


	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void copy_construct(Iterator out_dest_pointer_p, count_t dest_capacity_p, InputIterator source_pointer_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
		FE_ASSERT(dest_capacity_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(dest_capacity_p));
		FE_ASSERT(source_count_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_count_p));

		if constexpr (std::is_pointer<Iterator>::value == true)
		{
			FE::memcpy<DestAddressAlignment, SourceAddressAlignment>(	iterator_cast<T*>(out_dest_pointer_p), dest_capacity_p * sizeof(T), 
																		iterator_cast<T*>(source_pointer_p), source_count_p * sizeof(T));
		}
		else if constexpr (std::is_class<Iterator>::value == true)
		{
			if constexpr (std::is_same<typename Iterator::iterator_category, typename FE::contiguous_iterator<typename Iterator::value_type>::category>::value == true)
			{
				FE::memcpy<DestAddressAlignment, SourceAddressAlignment>(	iterator_cast<T*>(out_dest_pointer_p), dest_capacity_p * sizeof(T), 
																			iterator_cast<T*>(source_pointer_p), source_count_p * sizeof(T));
			}
			else
			{
				count_t l_max_count = MIN(dest_capacity_p, source_count_p);
				for (var::count_t i = 0; i < l_max_count; ++i)
				{
					*out_dest_pointer_p = *source_pointer_p;
					++out_dest_pointer_p;
					++source_pointer_p;
				}
			}
		}
	}
	
	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void copy_construct(Iterator out_dest_pointer_p, InputIterator source_pointer_p, count_t count_to_copy_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
		FE_ASSERT(count_to_copy_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(count_to_copy_p));

		if constexpr (std::is_pointer<Iterator>::value == true)
		{
			FE::memcpy<DestAddressAlignment, SourceAddressAlignment>(iterator_cast<T*>(out_dest_pointer_p), iterator_cast<T*>(source_pointer_p), count_to_copy_p * sizeof(T));
		}
		else if constexpr (std::is_class<Iterator>::value == true)
		{
			if constexpr (std::is_same<typename Iterator::iterator_category, typename FE::contiguous_iterator<typename Iterator::value_type>::category>::value == true)
			{
				FE::memcpy<DestAddressAlignment, SourceAddressAlignment>(iterator_cast<T*>(out_dest_pointer_p), iterator_cast<T*>(source_pointer_p), count_to_copy_p * sizeof(T));
			}
			else
			{
				for (var::count_t i = 0; i < count_to_copy_p; ++i)
				{
					*out_dest_pointer_p = *source_pointer_p;
					++out_dest_pointer_p;
					++source_pointer_p;
				}
			}
		}
	}


	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void move_construct(Iterator out_dest_pointer_p, count_t dest_capacity_p, InputIterator source_pointer_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
		FE_ASSERT(dest_capacity_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(dest_capacity_p));
		FE_ASSERT(source_count_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_count_p));

		memory_traits<T>::copy_construct<Iterator>(out_dest_pointer_p, dest_capacity_p, source_pointer_p, source_count_p);
	}

	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void move_construct(Iterator out_dest_pointer_p, InputIterator source_pointer_p, count_t count_to_copy_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
		FE_ASSERT(count_to_copy_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(count_to_copy_p));

		memory_traits<T>::copy_construct<Iterator>(out_dest_pointer_p, source_pointer_p, count_to_copy_p);
	}


	_FORCE_INLINE_ static void destruct(_MAYBE_UNUSED_ T& out_dest_p) noexcept
	{
		out_dest_p = T();
	}

	template<class Iterator>
	_FORCE_INLINE_ static void destruct(_MAYBE_UNUSED_ Iterator in_out_dest_first_p, _MAYBE_UNUSED_ Iterator in_out_dest_last_p) noexcept
	{
		FE_ASSERT(in_out_dest_first_p > in_out_dest_last_p, "${%s@0}: The begin iterator ${%s@1} must be pointing at the first element of a container.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_ILLEGAL_POSITION), TO_STRING(in_out_dest_first_p));

	}


	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void copy_assign(Iterator out_dest_pointer_p, count_t dest_capacity_p, InputIterator source_pointer_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
		FE_ASSERT(dest_capacity_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(dest_capacity_p));
		FE_ASSERT(source_count_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_count_p));

		memory_traits<T>::copy_construct<Iterator>(out_dest_pointer_p, dest_capacity_p, source_pointer_p, source_count_p);
	}
	
	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void copy_assign(Iterator out_dest_pointer_p, InputIterator source_pointer_p, count_t count_to_copy_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
		FE_ASSERT(count_to_copy_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(count_to_copy_p));

		memory_traits<T>::copy_construct<Iterator>(out_dest_pointer_p, source_pointer_p, count_to_copy_p);
	}


	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void move_assign(Iterator out_dest_pointer_p, count_t dest_count_p, InputIterator source_pointer_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
		FE_ASSERT(dest_count_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(dest_capacity_p));
		FE_ASSERT(source_count_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_count_p));

		memory_traits<T>::copy_construct<Iterator>(out_dest_pointer_p, dest_count_p, source_pointer_p, source_count_p);
	}

	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void move_assign(Iterator out_dest_pointer_p, InputIterator source_pointer_p, count_t count_to_copy_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
		FE_ASSERT(count_to_copy_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(count_to_copy_p));

		memory_traits<T>::copy_construct<Iterator>(out_dest_pointer_p, source_pointer_p, count_to_copy_p);
	}


	template<class Iterator>
	_FORCE_INLINE_ static void assign(Iterator in_out_dest_first_p, Iterator in_out_dest_last_p, const T& value_p) noexcept
	{
		static_assert(std::is_constructible<T>::value == true, "static assertion failed: The typename T must be copy constructible.");

		FE_ASSERT(in_out_dest_first_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(in_out_dest_last_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));

		FE_ASSERT(in_out_dest_first_p > in_out_dest_last_p, "${%s@0}: The begin iterator ${%s@1} must be pointing at the first element of a container.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_ILLEGAL_POSITION), TO_STRING(in_out_dest_first_p));

		memory_traits<T>::construct<Iterator>(in_out_dest_first_p, in_out_dest_last_p, value_p);
	}
};


template<typename T, ADDRESS DestAddressAlignment, ADDRESS SourceAddressAlignment>
class memory_traits<T, DestAddressAlignment, SourceAddressAlignment, TYPE_TRIVIALITY::_NOT_TRIVIAL> final
{
public:
	_MAYBE_UNUSED_ static constexpr inline TYPE_TRIVIALITY is_trivial = TYPE_TRIVIALITY::_NOT_TRIVIAL;


	_FORCE_INLINE_ static void construct(_MAYBE_UNUSED_ T& out_dest_p) noexcept
	{
		static_assert(std::is_constructible<T>::value == true, "static assertion failed: The typename T must be copy constructible.");

		new(&out_dest_p) T();
	}

	_FORCE_INLINE_ static void construct(T& out_dest_p, T value_p) noexcept
	{
		static_assert(std::is_constructible<T>::value == true, "static assertion failed: The typename T must be copy constructible.");

		new(&out_dest_p) T(std::move(value_p));
	}

	template<class Iterator>
	_FORCE_INLINE_ static void construct(Iterator in_out_dest_first_p, Iterator in_out_dest_last_p, const T& value_p) noexcept
	{
		static_assert(std::is_constructible<T>::value == true, "static assertion failed: The typename T must be copy constructible.");

		FE_ASSERT(in_out_dest_first_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(in_out_dest_last_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
		FE_ASSERT(in_out_dest_first_p > in_out_dest_last_p, "${%s@0}: The begin iterator ${%s@1} must be pointing at the first element of a container.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_ILLEGAL_POSITION), TO_STRING(in_out_dest_first_p));

		while (in_out_dest_first_p != in_out_dest_last_p)
		{
			new(iterator_cast<T*>(in_out_dest_first_p)) T(value_p);
			++in_out_dest_first_p;
		}
	}


	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void copy_construct(Iterator out_dest_pointer_p, count_t dest_capacity_p, InputIterator source_pointer_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));

		FE_ASSERT(dest_capacity_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(dest_capacity_p));
		FE_ASSERT(source_count_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_count_p));

		if (dest_capacity_p <= source_count_p)
		{
			for (var::count_t i = 0; i < dest_capacity_p; ++i)
			{
				new(iterator_cast<T*>(out_dest_pointer_p)) T(*source_pointer_p);
				++out_dest_pointer_p;
				++source_pointer_p;
			}
			return;
		}
		else
		{
			const T* const source_end = iterator_cast<const T* const>(source_pointer_p + source_count_p);
			while (source_pointer_p != source_end)
			{
				new(iterator_cast<T*>(out_dest_pointer_p)) T(*source_pointer_p);
				++out_dest_pointer_p;
				++source_pointer_p;
			}
			return;
		}
	}

	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void copy_construct(Iterator out_dest_pointer_p, InputIterator source_pointer_p, count_t count_to_copy_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));

		FE_ASSERT(count_to_copy_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(count_to_copy_p));

		for (var::count_t i = 0; i < count_to_copy_p; ++i)
		{
			new(iterator_cast<T*>(out_dest_pointer_p)) T(*source_pointer_p);
			++out_dest_pointer_p;
			++source_pointer_p;
		}
	}


	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void move_construct(Iterator out_dest_pointer_p, count_t dest_capacity_p, InputIterator source_pointer_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));

		FE_ASSERT(dest_capacity_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(dest_capacity_p));
		FE_ASSERT(source_count_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_count_p));

		if (dest_capacity_p <= source_count_p)
		{
			const T* const dest_end = iterator_cast<const T* const>(out_dest_pointer_p + dest_capacity_p);
			while (out_dest_pointer_p != dest_end)
			{
				new(iterator_cast<T*>(out_dest_pointer_p)) T(std::move(*source_pointer_p));
				++out_dest_pointer_p;
				++source_pointer_p;
			}
			return;
		}
		else
		{
			for (var::count_t i = 0; i < source_count_p; ++i)
			{
				new(iterator_cast<T*>(out_dest_pointer_p)) T(std::move(*source_pointer_p));
				++out_dest_pointer_p;
				++source_pointer_p;
			}
			return;
		}
	}

	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void move_construct(Iterator out_dest_pointer_p, InputIterator source_pointer_p, count_t count_to_copy_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));

		FE_ASSERT(count_to_copy_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(count_to_copy_p));

		for (var::count_t i = 0; i < count_to_copy_p; ++i)
		{
			new(iterator_cast<T*>(out_dest_pointer_p)) T(std::move(*source_pointer_p));
			++out_dest_pointer_p;
			++source_pointer_p;
		}
	}


	_FORCE_INLINE_ static void destruct(_MAYBE_UNUSED_ T& out_dest_p) noexcept
	{
		static_assert(std::is_destructible<T>::value == true, "static assertion failed: The typename T must be destructible.");
		out_dest_p.~T();
	}

	template<class Iterator>
	_FORCE_INLINE_ static void destruct(_MAYBE_UNUSED_ Iterator in_out_dest_first_p, Iterator in_out_dest_last_p) noexcept
	{
		static_assert(std::is_destructible<T>::value == true, "static assertion failed: The typename T must be destructible.");
		FE_ASSERT(in_out_dest_first_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(in_out_dest_first_p));
		FE_ASSERT(in_out_dest_last_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(in_out_dest_last_p));
		FE_ASSERT(in_out_dest_first_p > in_out_dest_last_p, "${%s@0}: The begin iterator ${%s@1} must be pointing at the first element of a container.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_ILLEGAL_POSITION), TO_STRING(in_out_dest_first_p));

		while (in_out_dest_first_p != in_out_dest_last_p)
		{
			in_out_dest_first_p->~T();
			++in_out_dest_first_p;
		}
	}


	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void copy_assign(Iterator out_dest_pointer_p, count_t dest_capacity_p, InputIterator source_pointer_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));

		FE_ASSERT(dest_capacity_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(dest_capacity_p));
		FE_ASSERT(source_count_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_count_p));

		if (dest_capacity_p <= source_count_p)
		{
			for (var::count_t i = 0; i < dest_capacity_p; ++i)
			{
				*out_dest_pointer_p = *source_pointer_p;
				++out_dest_pointer_p;
				++source_pointer_p;
			}
			return;
		}
		else
		{
			for (var::count_t i = 0; i < source_count_p; ++i)
			{
				*out_dest_pointer_p = *source_pointer_p;
				++out_dest_pointer_p;
				++source_pointer_p;
			}
			return;
		}
	}

	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void copy_assign(Iterator out_dest_pointer_p, InputIterator source_pointer_p, count_t count_to_copy_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));

		FE_ASSERT(count_to_copy_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(count_to_copy_p));

		for (var::count_t i = 0; i < count_to_copy_p; ++i)
		{
			*out_dest_pointer_p = *source_pointer_p;
			++out_dest_pointer_p;
			++source_pointer_p;
		}
	}


	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void move_assign(Iterator out_dest_pointer_p, count_t dest_capacity_p, InputIterator source_pointer_p, count_t source_count_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));

		FE_ASSERT(dest_capacity_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(dest_capacity_p));
		FE_ASSERT(source_count_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_count_p));

		if (dest_capacity_p <= source_count_p)
		{
			for (var::count_t i = 0; i < dest_capacity_p; ++i)
			{
				*out_dest_pointer_p = std::move(*source_pointer_p);
				++out_dest_pointer_p;
				++source_pointer_p;
			}
			return;
		}
		else
		{
			for (var::count_t i = 0; i < source_count_p; ++i)
			{
				*out_dest_pointer_p = std::move(*source_pointer_p);
				++out_dest_pointer_p;
				++source_pointer_p;
			}
			return;
		}
	}

	template<class Iterator, class InputIterator>
	_FORCE_INLINE_ static void move_assign(Iterator out_dest_pointer_p, InputIterator source_pointer_p, count_t count_to_copy_p) noexcept
	{
		FE_ASSERT(out_dest_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(source_pointer_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));

		FE_ASSERT(count_to_copy_p == 0, "${%s@0}: ${%s@1} is zero", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(count_to_copy_p));

		for (var::count_t i = 0; i < count_to_copy_p; ++i)
		{
			*out_dest_pointer_p = std::move(*source_pointer_p);
			++out_dest_pointer_p;
			++source_pointer_p;
		}
	}


	template<class Iterator>
	_FORCE_INLINE_ static void assign(Iterator in_out_dest_first_p, Iterator in_out_dest_last_p, const T& value_p) noexcept
	{
		static_assert(std::is_constructible<T>::value == true, "static assertion failed: The typename T must be copy constructible.");

		FE_ASSERT(in_out_dest_first_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(out_dest_pointer_p));
		FE_ASSERT(in_out_dest_last_p == nullptr, "${%s@0}: ${%s@1} is nullptr", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_NULLPTR), TO_STRING(source_pointer_p));
		FE_ASSERT(in_out_dest_first_p > in_out_dest_last_p, "${%s@0}: The begin iterator ${%s@1} must be pointing at the first element of a container.", TO_STRING(MEMORY_ERROR_1XX::_FATAL_ERROR_ILLEGAL_POSITION), TO_STRING(in_out_dest_first_p));

		while (in_out_dest_first_p != in_out_dest_last_p)
		{
			*in_out_dest_first_p = value_p;
			++in_out_dest_first_p;
		}
	}
};


END_NAMESPACE
#endif

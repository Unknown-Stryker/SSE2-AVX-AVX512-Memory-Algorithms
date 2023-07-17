// All Rights Reserved. 2023 by Hojin Lee (Unknown Stryker)
#include "memory.hpp"


BEGIN_NAMESPACE(FE)


#if _AVX512_ == true
void unaligned_memset_with_avx512(void* const dest_ptrc_p, int8 value_p, size_t total_bytes_p) noexcept
{
    FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
    FE_ASSERT(total_bytes_p == 0, "ERROR: element_bytes_p is 0.");

    __m512i* l_m512i_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
    const __m512i l_m512i_value_to_be_assigned = _mm512_set1_epi8(value_p);

    var::size_t l_leftover_bytes = MODULO_BY_64(total_bytes_p);
    size_t l_avx_operation_count = MODULO_BY_64(total_bytes_p - l_leftover_bytes);

    for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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

void aligned_memset_with_avx512(void* const dest_ptrc_p, int8 value_p, size_t total_bytes_p) noexcept
{
    FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
    FE_ASSERT(total_bytes_p == 0, "ERROR: element_bytes_p is 0.");
    FE_ASSERT((reinterpret_cast<uintptr_t>(dest_ptrc_p) % 64) != 0, "ERROR: dest_ptrc_p is not aligned by 64.");

    __m512i* l_m512i_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
    const __m512i l_m512i_value_to_be_assigned = _mm512_set1_epi8(value_p);

    var::size_t l_leftover_bytes = MODULO_BY_64(total_bytes_p);
    size_t l_avx_operation_count = MODULO_BY_64(total_bytes_p - l_leftover_bytes);

    for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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


void unaligned_memcpy_with_avx512(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
{
    FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
    FE_ASSERT(bytes_to_copy_p == 0, "ERROR: element_bytes_p is 0.");

    __m512i* l_m512i_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
    const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_ptrc_p);

    var::size_t l_leftover_bytes = MODULO_BY_64(bytes_to_copy_p);
    size_t l_avx_operation_count = MODULO_BY_64(bytes_to_copy_p - l_leftover_bytes);

    for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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

void aligned_memcpy_with_avx512(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
{
    FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
    FE_ASSERT(bytes_to_copy_p == 0, "ERROR: element_bytes_p is 0.");
    FE_ASSERT((reinterpret_cast<uintptr_t>(dest_ptrc_p) % 64) != 0, "ERROR: dest_ptrc_p is not aligned by 64.");
    FE_ASSERT((reinterpret_cast<uintptr_t>(source_ptrc_p) % 64) != 0, "ERROR: source_ptrc_p is not aligned by 64.");

    __m512i* l_m512i_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
    const __m512i* l_m512i_source_ptr = static_cast<const __m512i*>(source_ptrc_p);

    var::size_t l_leftover_bytes = MODULO_BY_64(bytes_to_copy_p);
    size_t l_avx_operation_count = MODULO_BY_64(bytes_to_copy_p - l_leftover_bytes);

    for (var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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
#elif _AVX_ == true
void unaligned_memset_with_avx(void* const dest_ptrc_p, int8 value_p, size_t total_bytes_p) noexcept
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

void aligned_memset_with_avx(void* const dest_ptrc_p, int8 value_p, size_t total_bytes_p) noexcept
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


void unaligned_memcpy_with_avx(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
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

void aligned_memcpy_with_avx(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
{
    FE_ASSERT(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
    FE_ASSERT(bytes_to_copy_p == 0, "ERROR: element_bytes_p is 0.");
    FE_ASSERT((reinterpret_cast<uintptr_t>(dest_ptrc_p) % 32) != 0, "ERROR: dest_ptrc_p is not aligned by 32.");
    FE_ASSERT((reinterpret_cast<uintptr_t>(source_ptrc_p) % 32) != 0, "ERROR: source_ptrc_p is not aligned by 32.");

    __m256i* l_m256i_dest_ptr = static_cast<__m256i*>(dest_ptrc_p);
    const __m256i* l_m256i_source_ptr = static_cast<const __m256i*>(source_ptrc_p);

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
#endif


void unaligned_memcpy(void* const dest_memblock_ptrc_p, length_t dest_length_p, size_t dest_element_bytes_p, const void* const source_memblock_ptrc_p, length_t source_length_p, size_t source_element_bytes_p) noexcept
{
    ABORT_IF(dest_memblock_ptrc_p == nullptr, "ERROR: dest_memblock_ptrc_p is nullptr.");
    ABORT_IF(source_memblock_ptrc_p == nullptr, "ERROR: source_memblock_ptrc_p is nullptr.");

    ABORT_IF(dest_length_p == 0, "ERROR: dest_length_p is 0.");
    ABORT_IF(dest_element_bytes_p == 0, "ERROR: dest_element_bytes_p is 0.");
    ABORT_IF(source_element_bytes_p == 0, "ERROR: source_element_bytes_p is 0.");

    size_t l_source_size = source_element_bytes_p * source_length_p;
    size_t l_dest_size = dest_element_bytes_p * dest_length_p;

    if (l_source_size >= l_dest_size)
    {
#if _AVX512_ == true
        ::FE::unaligned_memcpy_with_avx512(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_dest_size);
#elif _AVX_ == true
        ::FE::unaligned_memcpy_with_avx(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_dest_size);
#else
        ::memcpy(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_dest_size);
#endif
    }
    else
    {
#if _AVX512_ == true
        ::FE::unaligned_memcpy_with_avx512(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_source_size);
#elif _AVX_ == true
        ::FE::unaligned_memcpy_with_avx(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_source_size);
#else
        ::memcpy(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_source_size);
#endif
    }
}

void aligned_memcpy(void* const dest_memblock_ptrc_p, length_t dest_length_p, size_t dest_element_bytes_p, const void* const source_memblock_ptrc_p, length_t source_length_p, size_t source_element_bytes_p) noexcept
{
    ABORT_IF(dest_memblock_ptrc_p == nullptr, "ERROR: dest_memblock_ptrc_p is nullptr.");
    ABORT_IF(source_memblock_ptrc_p == nullptr, "ERROR: source_memblock_ptrc_p is nullptr.");

    ABORT_IF(dest_length_p == 0, "ERROR: dest_length_p is 0.");
    ABORT_IF(dest_element_bytes_p == 0, "ERROR: dest_element_bytes_p is 0.");
    ABORT_IF(source_element_bytes_p == 0, "ERROR: source_element_bytes_p is 0.");

    size_t l_source_size = source_element_bytes_p * source_length_p;
    size_t l_dest_size = dest_element_bytes_p * dest_length_p;

    if (l_source_size >= l_dest_size)
    {
#if _AVX512_ == true
        ::FE::aligned_memcpy_with_avx512(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_dest_size);
#elif _AVX_ == true
        ::FE::aligned_memcpy_with_avx(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_dest_size);
#else
        ::memcpy(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_dest_size);
#endif
    }
    else
    {
#if _AVX512_ == true
        ::FE::aligned_memcpy_with_avx512(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_source_size);
#elif _AVX_ == true
        ::FE::aligned_memcpy_with_avx(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_source_size);
#else
        ::memcpy(dest_memblock_ptrc_p, source_memblock_ptrc_p, l_source_size);
#endif
    }
}


END_NAMESPACE

#include "memory.hpp"


BEGIN_NAMESPACE(FE)


#if _AVX512_ == true
void unaligned_memset_with_avx512(void* const dest_ptrc_p, FE::int8 value_p, size_t bytes_to_set_p) noexcept
{
    __m512i* l_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
    const __m512i l_value_to_be_assigned = _mm512_set1_epi8(value_p);

    size_t l_leftover_bytes = MODULO_BY_32(bytes_to_set_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_set_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
    {
        _mm512_storeu_si512(l_dest_ptr, l_value_to_be_assigned);
        ++l_dest_ptr;
    }

    switch (l_leftover_bytes)
    {
    case 0:
        return;

    default:
        ::std::memset(l_dest_ptr, value_p, l_leftover_bytes);
        break;
    }
}

void aligned_memset_with_avx512(void* const dest_ptrc_p, FE::int8 value_p, size_t bytes_to_set_p) noexcept
{
    __m512i* l_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
    const __m512i l_value_to_be_assigned = _mm512_set1_epi8(value_p);

    size_t l_leftover_bytes = MODULO_BY_32(bytes_to_set_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_set_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
    {
        _mm512_store_si512(l_dest_ptr, l_value_to_be_assigned);
        ++l_dest_ptr;
    }

    switch (l_leftover_bytes)
    {
    case 0:
        return;

    default:
        ::std::memset(l_dest_ptr, value_p, l_leftover_bytes);
        break;
    }
}


void unaligned_memcpy_with_avx512(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
{
    __m512i* l_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
    const __m512i* l_source_ptr = static_cast<const __m512i* const>(source_ptrc_p);

    FE::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
    {
        _mm512_storeu_si512(l_dest_ptr, _mm512_loadu_si512(l_source_ptr));
        ++l_dest_ptr;
        ++l_source_ptr;
    }

    switch (l_leftover_bytes)
    {
    case 0:
        return;

    default:
        ::std::memcpy(l_dest_ptr, l_source_ptr, l_leftover_bytes);
        break;
    }
}

void aligned_memcpy_with_avx512(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
{
    __m512i* l_dest_ptr = static_cast<__m512i*>(dest_ptrc_p);
    const __m512i* l_source_ptr = static_cast<const __m512i* const>(source_ptrc_p);

    FE::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
    {
        _mm512_store_si512(l_dest_ptr, _mm512_load_si512(l_source_ptr));
        ++l_dest_ptr;
        ++l_source_ptr;
    }

    switch (l_leftover_bytes)
    {
    case 0:
        return;

    default:
        ::std::memcpy(l_dest_ptr, l_source_ptr, l_leftover_bytes);
        break;
    }
}
#elif _AVX_ == true
void unaligned_memset_with_avx(void* const dest_ptrc_p, FE::int8 value_p, size_t bytes_to_set_p) noexcept
{
    __m256i* l_dest_ptr = static_cast<__m256i*>(dest_ptrc_p);
    const __m256i l_value_to_be_assigned = _mm256_set1_epi8(value_p);

    size_t l_leftover_bytes = MODULO_BY_32(bytes_to_set_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_set_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
    {
        _mm256_storeu_si256(l_dest_ptr, l_value_to_be_assigned);
        ++l_dest_ptr;
    }

    switch (l_leftover_bytes)
    {
    case 0:
        return;

    default:
        ::std::memset(l_dest_ptr, value_p, l_leftover_bytes);
        break;
    }
}

void aligned_memset_with_avx(void* const dest_ptrc_p, FE::int8 value_p, size_t bytes_to_set_p) noexcept
{
    __m256i* l_dest_ptr = static_cast<__m256i*>(dest_ptrc_p);
    const __m256i l_value_to_be_assigned = _mm256_set1_epi8(value_p);

    size_t l_leftover_bytes = MODULO_BY_32(bytes_to_set_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_set_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
    {
        _mm256_store_si256(l_dest_ptr, l_value_to_be_assigned);
        ++l_dest_ptr;
    }

    switch (l_leftover_bytes)
    {
    case 0:
        return;

    default:
        ::std::memset(l_dest_ptr, value_p, l_leftover_bytes);
        break;
    }
}


void unaligned_memcpy_with_avx(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
{
    __m256i* l_dest_ptr = static_cast<__m256i*>(dest_ptrc_p);
    const __m256i* l_source_ptr = static_cast<const __m256i* const>(source_ptrc_p);

    FE::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
    {
        _mm256_storeu_si256(l_dest_ptr, _mm256_loadu_si256(l_source_ptr));
        ++l_dest_ptr;
        ++l_source_ptr;
    }

    switch (l_leftover_bytes)
    {
    case 0:
        return;

    default:
        ::std::memcpy(l_dest_ptr, l_source_ptr, l_leftover_bytes);
        break;
    }
}

void aligned_memcpy_with_avx(void* const dest_ptrc_p, const void* const source_ptrc_p, FE::size_t bytes_to_copy_p) noexcept
{
    __m256i* l_dest_ptr = static_cast<__m256i*>(dest_ptrc_p);
    const __m256i* l_source_ptr = static_cast<const __m256i* const>(source_ptrc_p);

    FE::size_t l_leftover_bytes = MODULO_BY_32(bytes_to_copy_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_copy_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 0; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
    {
        _mm256_store_si256(l_dest_ptr, _mm256_load_si256(l_source_ptr));
        ++l_dest_ptr;
        ++l_source_ptr;
    }

    switch (l_leftover_bytes)
    {
    case 0:
        return;

    default:
        ::std::memcpy(l_dest_ptr, l_source_ptr, l_leftover_bytes);
        break;
    }
}
#endif


void memset(void* const dest_ptrc_p, int8 value_p, size_t total_bytes_p) noexcept
{
    ABORT_IF(dest_ptrc_p == nullptr, "ERROR: dest_ptrc_p is nullptr.");
    ABORT_IF(total_bytes_p == 0, "ERROR: element_bytes_p is 0.");

#if _AVX512_ == true
    ::FE::unaligned_memset_with_avx512(dest_ptrc_p, value_p, total_bytes_p);
#elif _AVX_ == true
    ::FE::unaligned_memset_with_avx(dest_ptrc_p, value_p, total_bytes_p);
#else
    ::std::memset(dest_ptrc_p, value_p, total_bytes_p);
#endif
}

void memcpy_s(void* const dest_memblock_ptrc_p, length_t dest_length_p, size_t dest_element_bytes_p, const void* const source_memblock_ptrc_p, length_t source_length_p, size_t source_element_bytes_p) noexcept
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

END_NAMESPACE

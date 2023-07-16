#if _AVX512_ == true
void unaligned_memset_with_avx512(void* const dest_ptrc_p, FE::int8 value_p, size_t bytes_to_set_p) noexcept
{
    __m512i* l_dest_ptr = reinterpret_cast<__m512i*>(dest_ptrc_p);
    const __m512i l_value_to_be_assigned = _mm512_set1_epi8(value_p);

    size_t l_leftover_bytes = MODULO_BY_32(bytes_to_set_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_set_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 1; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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
    __m512i* l_dest_ptr = reinterpret_cast<__m512i*>(dest_ptrc_p);
    const __m512i l_value_to_be_assigned = _mm512_set1_epi8(value_p);

    size_t l_leftover_bytes = MODULO_BY_32(bytes_to_set_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_set_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 1; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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

    for (FE::var::size_t executed_operation_count = 1; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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

    for (FE::var::size_t executed_operation_count = 1; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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
    __m256i* l_dest_ptr = reinterpret_cast<__m256i*>(dest_ptrc_p);
    const __m256i l_value_to_be_assigned = _mm256_set1_epi8(value_p);

    size_t l_leftover_bytes = MODULO_BY_32(bytes_to_set_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_set_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 1; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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
    __m256i* l_dest_ptr = reinterpret_cast<__m256i*>(dest_ptrc_p);
    const __m256i l_value_to_be_assigned = _mm256_set1_epi8(value_p);

    size_t l_leftover_bytes = MODULO_BY_32(bytes_to_set_p);
    FE::var::size_t l_avx_operation_count = DIVIDE_BY_32(bytes_to_set_p - l_leftover_bytes);

    for (FE::var::size_t executed_operation_count = 1; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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

    for (FE::var::size_t executed_operation_count = 1; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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

    for (FE::var::size_t executed_operation_count = 1; executed_operation_count != l_avx_operation_count; ++executed_operation_count)
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

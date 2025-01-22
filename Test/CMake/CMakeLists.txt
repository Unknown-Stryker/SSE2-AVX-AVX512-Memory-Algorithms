# Copyright © from 2022 to present, UNKNOWN STRYKER. All Rights Reserved.
CMAKE_MINIMUM_REQUIRED(VERSION 3.25.0...3.31.2)
PROJECT(SIMD-Memory-Algorithms-Test
        VERSION 0.0.0
        DESCRIPTION "SIMD Memory Algorithms Tester.")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../Include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../Third-Party/benchmark-1.9.1/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../Third-Party/googletest-1.15.2/googletest/include)




IF(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND TARGET_CPU_ARCHITECTURE STREQUAL "x86-64")
    ENABLE_LANGUAGE(ASM_MASM)


ELSEIF(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND TARGET_CPU_ARCHITECTURE STREQUAL "arm64")
    ENABLE_LANGUAGE(ASM_MARMASM)


ELSEIF(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET_CPU_ARCHITECTURE STREQUAL "x86-64")
    #ENABLE_LANGUAGE(ASM-ATT)

    ADD_COMPILE_OPTIONS(-g -O3 -march=x86-64 -mavx -msse2)
    ADD_LINK_OPTIONS(-pthread -ldl)
    
    ADD_EXECUTABLE(${CMAKE_PROJECT_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        #${CMAKE_CURRENT_SOURCE_DIR}/../../Source/__x86_64_AVX_SSE_aligned_memcpy.asm
        #${CMAKE_CURRENT_SOURCE_DIR}/../../Source/__x86_64_AVX_SSE_dest_aligned_memcpy.asm
        #${CMAKE_CURRENT_SOURCE_DIR}/../../Source/__x86_64_AVX_SSE_source_aligned_memcpy.asm
        #${CMAKE_CURRENT_SOURCE_DIR}/../../Source/__x86_64_AVX_SSE_unaligned_memcpy.asm
    )
    
    TARGET_LINK_LIBRARIES(${CMAKE_PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Binaries/X86-64/Linux/libSIMD-Memory-Algorithms.a
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Third-Party/benchmark-1.9.1/lib/libbenchmark.a
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Third-Party/googletest-1.15.2/lib/libgtest.a
    )


ELSEIF(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET_CPU_ARCHITECTURE STREQUAL "arm64")


ELSE()
    MESSAGE(FATAL_ERROR "System not selected.")

ENDIF()
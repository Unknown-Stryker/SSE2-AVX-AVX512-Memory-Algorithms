# Copyright © from 2022 to present, UNKNOWN STRYKER. All Rights Reserved.
CMAKE_MINIMUM_REQUIRED(VERSION 3.25.0...3.31.2)
PROJECT(SIMD-Memory-Algorithms-Test
        VERSION 0.0.0
        DESCRIPTION "SIMD Memory Algorithms Tester.")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../Include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../Third-Party/benchmark-1.9.1/include)


ADD_COMPILE_OPTIONS(-g -O3 -std=c++20)
ADD_LINK_OPTIONS(-pthread -ldl)


ADD_EXECUTABLE(${CMAKE_PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)


TARGET_LINK_LIBRARIES(${CMAKE_PROJECT_NAME} 
${CMAKE_CURRENT_SOURCE_DIR}/../../Binaries/X86-64/Linux/libSIMD-Memory-Algorithms.a
${CMAKE_CURRENT_SOURCE_DIR}/../../Third-Party/benchmark-1.9.1/lib/libbenchmark.a
)
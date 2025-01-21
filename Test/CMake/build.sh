rm -r build
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_C_COMPILER=clang-19 -DCMAKE_SYSTEM_NAME="Linux" -DTARGET_CPU_ARCHITECTURE="x86-64" -G"Ninja"
cmake --build . -j8
cd ..
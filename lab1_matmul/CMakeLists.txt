cmake_minimum_required(VERSION 3.24)

project(lab1_matmul LANGUAGES CXX)

include(CheckLanguage)

option(MATMUL_USE_CUDA "Build with CUDA support when available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MATMUL_USE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(MATMUL_HAVE_CUDA TRUE)
  else()
    message(WARNING "CUDA compiler was not found. Building CPU-only fallback target.")
  endif()
endif()

add_executable(matmul
  src/main.cpp
  src/cpu.cpp
)

target_include_directories(matmul PRIVATE include)

target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

if(MATMUL_HAVE_CUDA)
  target_sources(matmul PRIVATE src/gpu.cu)
  target_compile_definitions(matmul PRIVATE MATMUL_HAVE_CUDA=1)
  set_target_properties(matmul PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
else()
  target_sources(matmul PRIVATE src/gpu_stub.cpp)
endif()

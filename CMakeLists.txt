# CMake minimum version
cmake_minimum_required(VERSION 3.20)

# Enable parallel compilation
set(CMAKE_BUILD_PARALLEL_LEVEL 1)

# Project name and languages (C++ and CUDA)
project(cachehit_llm CXX CUDA)

# Include CUDA configurations
include(cmake/cuda.cmake)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard to CUDA 14
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Option for LLAMA3 support
option(LLAMA3_SUPPORT OFF)
if(LLAMA3_SUPPORT)
  message(STATUS "LLAMA3_SUPPORT ON")
  add_definitions(-DLLAMA3_SUPPORT)
endif()
# Option for QWEN2 support
option(QWEN2_SUPPORT OFF)
if(QWEN2_SUPPORT)
  message(STATUS "QWEN2_SUPPORT ON")
  add_definitions(-DQWEN2_SUPPORT)
endif()

# Find required third-party libraries
find_package(GTest REQUIRED)      # Google Test
find_package(glog REQUIRED)       # Google Logging (glog)
# find_package(Armadillo REQUIRED)  # Armadillo

# Manually set Armadillo paths if not found
if (NOT ARMADILLO_FOUND)
    message(WARNING "Armadillo not found, manually setting paths...")
    set(ARMADILLO_INCLUDE_DIRS "/home/tonglei/armadillo_local_install/include")
    set(ARMADILLO_LIBRARIES "/home/tonglei/armadillo_local_install/lib/libarmadillo.so")
endif()

# Collect source files from frontend directories
aux_source_directory(cachehitML/src/base         BASE_DIR)
aux_source_directory(cachehitML/src/tensor       TENSOR_DIR)
aux_source_directory(cachehitML/src/op           OP_DIR)
aux_source_directory(cachehitML/src/op/backend   OP_BACKEND_DIR)
aux_source_directory(cachehitML/src/model        MODEL_DIR)
aux_source_directory(cachehitML/src/sampler      SAMPLER_DIR)

# Collect source files from backend directories
aux_source_directory(cachehitML/src/op/backend/cuda-backend   OP_CUDA_BACKEND_DIR)
aux_source_directory(cachehitML/src/op/backend/x86cpu-backend OP_X86_BACKEND_DIR)

# Set library output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Create shared library "llama"
add_library(llama SHARED
  ${BASE_DIR}
  ${TENSOR_DIR}
  ${OP_DIR}
  ${OP_BACKEND_DIR}
  ${MODEL_DIR}
  ${SAMPLER_DIR}
  ${OP_CUDA_BACKEND_DIR}
  ${OP_X86_BACKEND_DIR}
)

# Link libraries
target_link_libraries(llama 
  # Armadillo               # Armadillo
  # sentencepiece           # SentencePiece
  glog::glog              # Google Logging (glog)
  # gtest gtest_main        # Google Test  pthread                 # pthread
  cudart                  # CUDA runtime
)

# Add CUDA library directories
target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

# Include directories
target_include_directories(llama PUBLIC
  ${ARMADILLO_INCLUDE_DIRS}  # Armadillo headers
  ${glog_INCLUDE_DIR}        # Google Logging (glog) headers
  ${PROJECT_SOURCE_DIR}/cachehitML/include  # Project headers
  ${PROJECT_SOURCE_DIR}/cachehitML/src/op   # op headers
  ${PROJECT_SOURCE_DIR}/cachehitML/src/op/backend # op/backend headers
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}  # CUDA headers
)

# Enable CUDA separable compilation
set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add subdirectories
add_subdirectory(test)  # Add "test" subdirectory
# add_subdirectory(demo)  # Uncomment if "demo" directory is ready
#include <base/alloc.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <iostream>
#include <cstring>

// CPU malloc and release part
TEST(CPUAllocatorTest, ML_malloc_test) {
  cachehitML::CPUAllocator allocator;
  // 0 byte == nullptr
  void* ptr = allocator.ML_malloc(0);
  ASSERT_EQ(ptr, nullptr) << "0 byte == nullptr";
  // 1byte == 64byte
  ptr = allocator.ML_malloc(1);
  ASSERT_NE(ptr, nullptr);
  allocator.ML_release(ptr);
}

TEST(CPUAllocatorTest, ML_release_test) {
  cachehitML::CPUAllocator allocator;
  void* ptr = allocator.ML_malloc(32);
  ASSERT_NE(ptr, nullptr);
  allocator.ML_release(ptr);
}

TEST(CPUAllocatorTest, ML_release_nullptr_test) {
  cachehitML::CPUAllocator allocator;
  allocator.ML_release(nullptr);
}

TEST(CUDAAllocatorTest, ML_malloc_test) {
  cachehitML::CUDAAllocator allocator;
  void* ptr = allocator.ML_malloc(32);
  ASSERT_NE(ptr, nullptr);

  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
  ASSERT_EQ(err, cudaSuccess);
  ASSERT_EQ(attributes.type, cudaMemoryTypeDevice)
      << "ptr is not device memory";
}

// CPU or CUDA Memcpy part
TEST(Memcpy_test, ML_memcpy_cpu2cpu) {
  cachehitML::CPUAllocator allocator;
  uint32_t size = 32;
  void* ptr = allocator.ML_malloc(size);
  ASSERT_NE(ptr, nullptr);
  std::memset(ptr, 0, size);
  for (int i = 0; i < size; i++) {
    ((uint8_t*)ptr)[i] = i;
  }
  void* ptr1 = allocator.ML_malloc(size);
  allocator.ML_memcpy(ptr1, ptr, size, cachehitML::MemcpyKind::kMemcpyCPU2CPU,
                      nullptr, false);
  ASSERT_NE(ptr1, nullptr);
}

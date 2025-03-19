#include <glog/logging.h>

#include <cstdlib>

#include "base/alloc.h"

#if __STDC_VERSION__ >= 201112L
#define CACHEHIT_HAVE_ALIGNED_ALLOC
#endif

namespace cachehitML {

CPUAllocator::CPUAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

void *CPUAllocator::ML_malloc(size_t byte_size) const {
  if (byte_size <= 0) {
    return nullptr;
  }
  void *ptr = nullptr;
#ifdef CACHEHIT_HAVE_ALIGNED_ALLOC
  const size_t alignment =
      (byte_size >= size_t(1024)) ? size_t(128) : size_t(64);
  ptr = std::aligned_alloc(alignment, byte_size);
#else
  ptr = std::malloc(byte_size);
#endif
  CHECK(ptr != nullptr) << "Failed to allocate " << byte_size << " bytes";
  return ptr;
}

void CPUAllocator::ML_release(void *ptr) const {
  if (ptr != nullptr) {
    std::free(ptr);
  }
}

}  // namespace cachehitML

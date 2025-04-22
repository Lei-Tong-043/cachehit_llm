#ifndef CACHEHIT_INCLUDE_BASE_ALLOC_H_
#define CACHEHIT_INCLUDE_BASE_ALLOC_H_

#include <map>
#include <memory>

#include "base.h"

namespace cachehitML {

enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};

// device mem allocator
// all device allocator base on this abstact class
class DeviceAllocator {
 public:
  // avoid implicit conversion by other type of data
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

  virtual DeviceType device_type() const { return device_type_; }

  virtual void ML_release(void *ptr) const = 0;

  virtual void *ML_malloc(size_t byte_size) const = 0;

  virtual void ML_memcpy(void *dst, const void *src, size_t byte_size,
                         MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,
                         void *stream = nullptr, bool need_sync = false) const;

  virtual void ML_memset_zero(void *ptr, size_t byte_size, void *stream, bool need_sync = false);

  // virtual ~DeviceAllocator() = default;

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

// CPU mem allocator, inherit from DeviceAllocator
class CPUAllocator : public DeviceAllocator {
 public:
  explicit CPUAllocator();

  void ML_release(void *ptr) const override;

  void *ML_malloc(size_t byte_size) const override;
};

// cuda memory buffer strcut
// record data_ptr, byte_size, busy flag
struct CudaMemoryBuffer {
  void *data;
  size_t byte_size;
  bool busy;

  CudaMemoryBuffer() = default;

  CudaMemoryBuffer(void *data, size_t byte_size, bool busy)
      : data(data), byte_size(byte_size), busy(busy) {}
};

// CUDA mem allocator, inherit from DeviceAllocator
// memory pool :
// 1. small part, each mem <= 1M(byte) in cuda_buffers_map_
// 2. big part, each mem > 1M(byte) in big_buffers_map_
class CUDAAllocator : public DeviceAllocator {
 public:
  explicit CUDAAllocator();

  void ML_release(void *ptr) const override;

  void *ML_malloc(size_t byte_size) const override;

 private:
  mutable std::map<int, size_t> no_busy_cnt_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

// CPU mem alloc factory(Singleton pattern)
// ensure only one CPUAllocator instance
class CPUAllocatorFactory {
 public:
  static std::shared_ptr<CPUAllocator> get_instance() {
    if (instance_ == nullptr) {
      instance_ = std::make_shared<CPUAllocator>();
    }
    return instance_;
  }

 private:
  static std::shared_ptr<CPUAllocator> instance_;
};

// CUDA mem alloc factory(Singleton pattern)
// ensure only one CUDAAllocator instance
class CUDAAllocatorFactory {
 public:
  static std::shared_ptr<CUDAAllocator> get_instance() {
    if (instance_ == nullptr) {
      instance_ = std::make_shared<CUDAAllocator>();
    }
    return instance_;
  }

 private:
  static std::shared_ptr<CUDAAllocator> instance_;
};

}  // namespace cachehitML
#endif  // CACHEHIT_INCLUDE_BASE_ALLOC_H_
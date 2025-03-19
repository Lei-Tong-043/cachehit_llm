#pragma once

#include <base/alloc.h>

#include <memory>

namespace cachehitML {
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 public:
  explicit Buffer() = default;
  explicit Buffer(size_t byte_size,
                  std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);
  virtual ~Buffer();

  bool allocate();

  void copy_from(const Buffer& other) const;
  void copy_from(const Buffer* other) const;

  void* ptr();
  const void* ptr() const;

  size_t byte_size() const;

  std::shared_ptr<DeviceAllocator> allocator() const;

  DeviceType device_type() const;

  void set_device_type(DeviceType device_type);

  std::shared_ptr<Buffer> get_shared_from_this(); 

  bool is_external() const;

 private:
  size_t byte_size_ = 0;
  std::shared_ptr<DeviceAllocator> allocator_;
  void* ptr_ = nullptr;
  bool use_external_ = false;
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};
}  // namespace cachehitML
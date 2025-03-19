#pragma once

#include "layer.h"

namespace op {
class RoPELayer : public Layer {
  explicit RoPELayer(cachehitML::DeviceType device_type, int32_t dim, int32_t kv_dim,
                     int32_t head_size);

  cachehitML::Status check() const override;

  cachehitML::Status forward() override;

 private:
  int32_t dim_ = 0;
  int32_t kv_dim_ = 0;
  int32_t head_size_ = 0;
};
}  // namespace op
#pragma once
#include "op_layer.h"
namespace op {
class SwiGLULayer : public op::Layer {
 public:
  explicit SwiGLULayer(cachehitML::DeviceType device_type, int32_t hidden_dim);

  cachehitML::Status check() const override;
  cachehitML::Status forward() override;

 private:
  int32_t hidden_dim_ = 0;
};
}  // namespace op
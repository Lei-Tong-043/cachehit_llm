#pragma once
#include "layer.h"

namespace op {
class RmsNormLayer : public Layer {
 public:
  explicit RmsNormLayer(cachehitML::DeviceType device_type, int32_t dim);

  cachehitML::Status check() const override;

  cachehitML::Status forward() override;

 private:
  int32_t dim_ = 0;
};
}  // namespace op
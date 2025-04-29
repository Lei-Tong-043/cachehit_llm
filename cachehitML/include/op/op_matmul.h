#pragma once

#include "base/cuda_config.h"
#include "op_layer.h"

namespace op {
class MatmulLayer : public Layer {
 public:
  explicit MatmulLayer(cachehitML::DeviceType device_type, int32_t dim0, int32_t dim1,
                       bool is_quant_layer = false, bool has_bias = false);
  cachehitML::Status check() const override;

  cachehitML::Status forward() override;

  cachehitML::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                              cachehitML::DeviceType device_type);

  tensor::Tensor& get_bias(int32_t idx);

  const tensor::Tensor& get_bias(int32_t idx) const;

  void to_cuda() override;

 private:
  int32_t dim0_;
  int32_t dim1_;
  bool has_bias_ = false;
  std::vector<tensor::Tensor> inputs_;
};
}  // namespace op
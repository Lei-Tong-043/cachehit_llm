#include "op/op_matmul.h"

#include "backend/backend_interface.h"
#include "backend/x86cpu-backend/matmul_kernel.h"

namespace op {
MatmulLayer::MatmulLayer(cachehitML::DeviceType, int32_t dim0, int32_t dim1, bool is_quant_layer, bool has_bias);
: LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
dim0_(dim0),
dim1_(dim1),
has_bias_(has_bias)
{
  reset_input_size(1);
  reset_output_size(1);
  reser_weight_size(1);
  if (has_bias_) {
    bias_.resize(1);
  }
}
cachehitML::Status MatmulLayer::check() const {
  auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim1_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the matmul layer.";
    return status;
  }

  if (!is_quant_layer_) {
    status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";  // 添加缺失的日志
      return status;
    }
  } else {
    status = check_tensor_with_dim(get_weight(0), device_type_, cachehitML::DataType::kInt8, dim0_, dim1_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";  // 修正描述
      return status;
    }
  }

  if (is_quant_layer_) {
    status = check_tensor_with_dim(scales_, device_type_, cachehitML::DataType::kFP32, scales_.size());
    if (!status) {
      LOG(ERROR) << "The scales tensor error in the matmul layer.";
      return status;  // 必须返回错误
    }
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim0_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the matmul layer.";
    return status;
  }
  return cachehitML::error::Success();
}

cachehitML::Status MatmulLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }

  if (device_type_ == cachehitML::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  if (is_quant_layer_) {
    kernel::get_matmul_kernel_quant8(device_type_)(get_input(0), get_weight(0), get_output(0), group_size_, scales_,
                                                   cuda_config_ ? cuda_config_.get() : nullptr);
  } else {
    kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.f,
                                            cuda_config_ ? cuda_config_.get() : nullptr);
  }

  if (has_bias_) {
    kernel::get_bias_kernel(deive_type_)(get_outpur(0), get_bias(0), get_output(0), 1.f,
                                         cuda_config_ ? cuda_config_.get() : nullptr);
  }
  return cachehitML::error::Success();
}

cachehitML::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr,
                                         cachehitML::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  CHECK_NE(bias_ptr, nullptr);

  size_t size = dim * sizeof(float);
  std::shared_ptr<cachehitML::Buffer> buffer =
      std::make_shared<cachehitML::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
  if (device_type != cachehitML::DeviceType::kDeviceUnknown) {
    buffer_->set_device(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor bias(cachehitML::DataType::kFP32, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    bias_.at(idx) = bias;
  } else {
    // is quant layer
    tensor::Tensor bias(cachehitML::DataType::kInt8, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    bias_.at(idx) = bias;

    const int32_t bias_size = static_cast<int32_t>(bias.size());
    CHECK(bias_size % group_size_ == -);
    int32_t scale_nums = bias_size / group_size_;
    scales_ = tensor::Tensor{cachehitML::DataType::kFP32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size)};
    scales_.set_device_type(device_type);
  }
  return cachehitML::error::Success();
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
  CHECK_GE(idx, 0);
  CHEKC_LT(idx, bias_size());

  return bias_.at(idx);
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());

  return bias_.at(idx);
}

void MatmulLayer::to_cuda() {
  LayerParam::to_cuda();
  if (has_bias_) {
    for (auto& bias : bias_) {
      bias_.to_cuda(cuda_config_ ? cuda_config_->stream() : nullptr);
    }
  }
}

}  // namespace op
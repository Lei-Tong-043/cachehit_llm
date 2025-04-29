#include "op/op_layer.h"

#include <glog/logging.h>

#include <cstdarg>
#include <numeric>
#include <utility>

#include "base/cuda_config.h"

namespace op {
///
///
/// BaseLayer Impl
BaseLayer::BaseLayer(cachehitML::DeviceType device_type, LayerType layer_type,
                     cachehitML::DataType data_type, std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {}

cachehitML::DataType BaseLayer::data_type() const { return data_type_; }

LayerType BaseLayer::layer_type() const { return layer_type_; }

cachehitML::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
  return cachehitML::error::FuntionUnImplement();
}

cachehitML::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                         const void* ptr, cachehitML::DeviceType device_type) {
  return cachehitML::error::FuntionUnImplement();
}

const std::string& BaseLayer::get_layer_name() const { return layer_name_; }

void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }

cachehitML::DeviceType BaseLayer::device_type() const { return device_type_; }

void BaseLayer::set_device_type(cachehitML::DeviceType device_type) { device_type_ = device_type; }

///
///
/// Layer Impl
Layer::Layer(cachehitML::DeviceType device_type, LayerType layer_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, cachehitML::DataType::kFP32, std::move(layer_name)) {}

cachehitML::Status Layer::init() { return cachehitML::error::Success(); }

cachehitML::Status Layer::forward() { return cachehitML::error::FuntionUnImplement(); }

cachehitML::Status Layer::check_tensor(const tensor::Tensor& tensor,
                                       cachehitML::DeviceType device_type,
                                       cachehitML::DataType data_type) const {
  if (tensor.is_empty()) {
    return cachehitML::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return cachehitML::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return cachehitML::error::InvalidArgument("The tensor has a wrong data type.");
  }
  return cachehitML::error::Success();
}

cachehitML::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor,
                                                cachehitML::DeviceType device_type,
                                                cachehitML::DataType data_type, ...) const {
  std::va_list args;
  if (tensor.is_empty()) {
    return cachehitML::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return cachehitML::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return cachehitML::error::InvalidArgument("The tensor has a wrong data type.");
  }
  va_start(args, data_type);
  int32_t dims = tensor.dims_size();
  for (int32_t i = 0; i < dims; ++i) {
    int32_t dim = va_arg(args, int32_t);
    if (tensor.get_dim(i) != dim) {
      return cachehitML::error::InvalidArgument("The tensor has a wrong dimension." +
                                                std::to_string(i));
    }
  }
  va_end(args);
  return cachehitML::error::Success();
}

cachehitML::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->forward();
}

cachehitML::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                  const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);

  this->set_output(0, output1);
  return this->forward();
}

cachehitML::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                  const tensor::Tensor& input3, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);

  this->set_output(0, output1);
  return this->forward();
}

cachehitML::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                  const tensor::Tensor& input3, const tensor::Tensor& input4,
                                  const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);

  this->set_output(0, output1);
  return this->forward();
}

cachehitML::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                  const tensor::Tensor& input3, const tensor::Tensor& input4,
                                  const tensor::Tensor& input5, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);

  this->set_output(0, output1);
  return this->forward();
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->inputs_.at(idx) = output;
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}
const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

tensor::Tensor& Layer::get_input(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_output(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

cachehitML::Status Layer::check() const {
  return cachehitML::error::FuntionUnImplement("The check function is not implement yet");
}

void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

void Layer::reset_output_size(size_t size) { outputs_.resize(size); }

void Layer::to_cuda() {
  for (auto& input : inputs_) {
    if (!input.is_empty()) {
      input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
  for (auto& output : outputs_) {
    if (!output.is_empty()) {
      output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
  if (!config) {
    return;
  }
  this->cuda_config_ = config;
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const { return cuda_config_; }

size_t Layer::input_size() const { return inputs_.size(); }

size_t Layer::output_size() const { return outputs_.size(); }

///
/// Layer with weight
/// LayerParam Impl

LayerParam::LayerParam(cachehitML::DeviceType device_type, LayerType layer_type,
                       bool is_quant_layer, std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)), is_quant_layer_(is_quant_layer) {}

cachehitML::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == cachehitML::DataType::kFP32);
  if (!weight.is_empty()) {
    CHECK(weight.device_type() == device_type_);
  }
  weights_.at(idx) = weight;
  return cachehitML::error::Success();
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

tensor::Tensor& LayerParam::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

void LayerParam::to_cuda() {
  Layer::to_cuda();
  for (auto& weight : weights_) {
    weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
  if (!scales_.is_empty()) {
    scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
}

cachehitML::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                          const void* weight_ptr,
                                          cachehitML::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK_NE(weight_ptr, nullptr);

  size_t size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
  std::shared_ptr<cachehitML::Buffer> buffer =
      std::make_shared<cachehitML::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != cachehitML::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor weight(cachehitML::DataType::kFP32, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;
  } else {
    // is quant layer
    // only support int8 quant layer
    tensor::Tensor weight(cachehitML::DataType::kInt8, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;

    const int32_t weight_size = static_cast<int32_t>(weight.size());
    CHECK(weight_size % group_size_ == 0);

    int32_t scale_nums = weight_size / group_size_;
    scales_ = tensor::Tensor{cachehitML::DataType::kFP32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
    scales_.set_device_type(device_type);
  }

  return cachehitML::error::Success();
}

void LayerParam::set_scales(const tensor::Tensor& scales) {
  CHECK(!scales.is_empty());
  this->scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size) { this->group_size_ = group_size; }

int32_t LayerParam::get_scale_num() const {
  CHECK(!scales_.is_empty());
  return static_cast<int32_t>(scales_.size());
}

void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

size_t LayerParam::weight_size() const { return weights_.size(); }

}  // namespace op

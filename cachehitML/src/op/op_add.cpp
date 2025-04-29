#include "op/op_add.h"

#include "backend/backend_interface.h"

namespace op {
VecAddLayer::VecAddLayer(cachehitML::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "Add")

{
  reset_input_size(2);
  reset_output_size(1);
}

cachehitML::Status VecAddLayer::check() const {
  tensor::Tensor input1 = this->get_input(0);
  tensor::Tensor input2 = this->get_input(1);
  int32_t size = input1.size();
  cachehitML::Status status;
  status = check_tensor_with_dim(input1, device_type_, data_type_, size);
  if (!status) {
    LOG(ERROR) << "add kernel input tensor 1 error.";
    return status;
  }

  status = check_tensor_with_dim(input2, device_type_, data_type_, size);
  if (!status) {
    LOG(ERROR) << "add kernel input tensor 2 error.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, size);
  if (!status) {
    LOG(ERROR) << "add kernel output tensor 1 error.";
    return status;
  }

  return cachehitML::error::Success();
}

cachehitML::Status VecAddLayer::forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);

  if (device_type_ == cachehitML::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_add_kernel(device_type_)(input1, input2, output,
                                      cuda_config_ ? cuda_config_->stream : nullptr);

  return cachehitML::error::Success();
}

}  // namespace op

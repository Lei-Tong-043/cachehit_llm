// #include "op/add.h"

// #include "backend/backend_interface.h"

// namespace op {
// namespace ML = cachehitML;

// VecAddLayer::VecAddLayer(ML::DeviceType device_type)
//     : Layer(device_type, LayerType::kLayerAdd, "Add") {
//   reset_input_size(2);
//   reset_output_size(1);
// }

// ML::Status VecAddLayer::check() const {
//   tensor::Tensor input1 = this->get_input(0);
//   tensor::Tensor input2 = this->get_input(1);

//   int32_t size = input1.size();
//   ML::Status status;
//   status = check_tensor_with_dim(input1, device_type_, data_type_, size);
//   if (!status) {
//     LOG(ERROR) << "The input tensor 1 error in add layer.";
//     return status;
//   }

//   status = check_tensor_with_dim(input2, device_type_, data_type_, size);
//   if (!status) {
//     LOG(ERROR) << "The input tensor 2 error in add layer.";
//     return status;
//   }

//   status = check_tensor_with_dim(get_output(0), device_type_, data_type_, size);
//   if (!status) {
//     LOG(ERROR) << "The output tensor error in add layer.";
//     return status;
//   }
//   return ML::error::Success();
// }

// ML::Status VecAddLayer::forward() {
//   auto status = this->check();
//   if (!status) {
//     return status;
//   }
//   auto input1 = this->get_input(0);
//   auto input2 = this->get_input(1);
//   auto output = this->get_output(0);

//   if (device_type_ == ML::DeviceType::kDeviceCUDA) {
//     CHECK(cuda_config_ != nullptr);
//   }
//   kernel::get_add_kernel(device_type_)(input1, input2, output,
//                                        cuda_config_ ? cuda_config_->stream : nullptr);
//   return ML::error::Success();
// }

// }  // namespace op
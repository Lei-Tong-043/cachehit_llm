#include "backend/backend_interface.h"
#include "op/op_embedding.h"
#include "backend/x86cpu-backend/emb_kernel.h"
#include "op/op_layer.h"
namespace op {
EmbeddingLayer::EmbeddingLayer(cachehitML::DeviceType device_type, int32_t dim, int32_t seq_len,
                               int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding") {
  reset_weight_size(1);
  reset_input_size(2);
  reset_output_size(1);
}

cachehitML::Status EmbeddingLayer::check() const{
  const auto& input_tensor = get_input(0);
  const auto& token_size = get_input(1).size();
  if (token_size > input_tensor.size()) {
    return cachehitML::error::InvalidArgument("token_size should be less than input_tensor size");
  }

  cachehitML::Status status = check_tensor_with_dim(
      input_tensor, cachehitML::DeviceType::kDeviceCPU, cachehitML::DataType::kFP32, token_size);
  if (!status) {
    LOG(ERROR) << "The input tensor errorr in the embedding layer.";
    return status;
  }

  status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, vocab_size_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the embedding layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, token_size, dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the embedding layer.";
    return status;
  }

  return cachehitML::error::Success();
}

cachehitML::Status EmbeddingLayer::forward() {
  cachehitML::Status status = check();
  if (!status) {
    return status;
  }
  if (device_type_ == cachehitML::DeviceType::kDeviceCPU) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), vocab_size_,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  
  return cachehitML::error::Success();
}
}  // namespace op
#ifndef CACHEHITML_INCLUDE_OP_LAYER_H_
#define CACHEHITML_INCLUDE_OP_LAYER_H_

#include <base/cuda_config.h>
#include <string>
#include <vector>
#include <base/base.h>
#include <tensor/tensor.h>

namespace op{
using ML = cachehitML;
class Layer;
enum class LayerType : uint8_t{
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding,
  kLayerRMSNorm,
  kLayerMatmul,
  kLayerRoPE,
  kLayerMHA,
  kLayerSoftmax,
  kLayerAdd,
  kLayerSwiGLU,
};

class BaseLayer{
public:
  explicit BaseLayer(ML::DeviceType device_type, LayerType layer_type, ML::DataType data_type,
                     std::string layer_name = "");
  
  ML::DataType data_type() const;
  
  LayerType layer_type() cosnt;
  // 
  virtual ML::Status init() = 0;
  // 
  virtual ML::Status forward() = 0;
  //
  virtual ML::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;
  //
  virtual ML::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& output1) = 0;
  //
  virtual ML::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;
  // 
  virtual ML::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& input3, const tensor::Tensor& input4,
                             const tensor::Tensor& output1) = 0;
  //                             
  virtual ML::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& input3, const tensor::Tensor& input4,
                             const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;  
  // 
  virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;
  
  virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

  virtual size_t input_size() const = 0;

  virtual size_t output_size() const = 0;

  virtual ML::Status check() const = 0;

  virtual tensor::Tensor& get_input(int32_t idx) = 0;

  virtual tensor::Tensor& get_output(int32_t idx) = 0;

  virtual const tensor::Tensor& get_input(int32_t idx) = 0;

  virtual const tensor::Tensor& get_output(int32_t idx) = 0;

  virtual ML::Status set_weight(int32_t idx, const tensor::Tensor& weight);

  virtual ML::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                const void* ptr,
                                ML::DeviceType device_type = ML::DeviceType::DEVICE_UNKNOWN);
  const std::string& get_layer_name() const;

  void set_layer_name(const std::string& layer_name);

  ML::DeviceType device_type() const;

  void set_device_type(ML::DeviceType device_type);

protected:
  std::string layer_name_;
  LayerType layer_type_ = LayerType::kLayerUnknown;
  ML::DataType data_type_ = DATA_TYPE_UNKNOWN;
  ML::DeviceType device_type_ = DEVICE_UNKNOWN;
};

class Layer : public BaseLayer {
public:
  explicit Layer(ML::DeviceType device_type, LayerType layer_type, std::string layer_name = "");

  ML::Status init() override;

  ML::Status check_tensor(const tensor::Tensor& tensor, ML::DeviceType device_type,
                          ML::DataType data_type) const;

  ML::Status check_tensor_with_dim(const tensor::Tensor& tensor, ML::DeviceType device_type,
                                   ML::DataType data_type, ...);
  ML::Status check() const override;

  ML::Status forward() override;

  ML::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;
  //
  ML::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                     const tensor::Tensor& output1) override;
  //
  ML::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                     const tensor::Tensor& input3, const tensor::Tensor& output1) override;
  // 
  ML::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                     const tensor::Tensor& input3, const tensor::Tensor& input4,
                     const tensor::Tensor& output1) override;
  //                             
  ML::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                     const tensor::Tensor& input3, const tensor::Tensor& input4,
                     const tensor::Tensor& input5, const tensor::Tensor& output1) override;

  void set_input(int32_t idx, const tensor::Tensor& input) override;

  void set_output(int32_t idx, const tensor::Tensor& output) override;

  const tensor::Tensor& get_input(int32_t idx) const override;

  const tensor::Tensor& get_output(int32_t idx) const override;

  tensor::Tensor& get_input(int32_t idx) override;

  tensor::Tensor& get_output(int32_t idx) override;

  size_t input_size() const override;

  size_t output_size() const override;

  void reset_input_size(size_t size);

  void reset_output_size(size_t size);

  virtual void to_cuda();

  void set_cuda_config(std::shared_ptr<kernel::CudaConfig> cuda_config);
  
  std::shared_ptr<kernel::CudaConfig> cuda_config() const;

protected:
  std::vector<tensor::Tensor> inputs_;
  std::vector<tensor::Tensor> outputs_;
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
};

class LayerParam : public Layer{
  explicit LayerParam(ML::DeviceType device_type, LayerType layer_type,
                      bool is_quant_layer = false, std::string layer_name = "");

  size_t weight_size() const;

  void reset_weight_size(size_t size);

  tensor::Tensor& get_weigth(size_t size);

  const tensor::Tensor& get_weight(int32_t idx) const;

  void to_cuda() override;

  ML::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

  ML::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
                        ML::DeviceType device_type = ML::DeviceType::DEVICE_UNKNOWN) override;
  void set_scales(const tensor::Tensor& scales);
  
  void set_group_size(int32_t group_size);

  int32_t get_scale_num() const;

protected:
  int32_t group_size_ = 0;
  bool is_quant_layer_ = false;
  tensor::Tensor scales_;
  std::vector<tensor::Tensor> weights_;
}

}

#endif
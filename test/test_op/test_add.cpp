#include <glog/logging.h>
#include <gtest/gtest.h>

#include <iostream>
#include <string>

#include "backend/backend_interface.h"
#include "op/op_add.h"
#include "tensor/tensor.h"

using namespace cachehitML;
TEST(VecAddTest, cpu) {
  int32_t m = 1000;
  auto dtype = DataType::kFP32;
  auto device_type = DeviceType::kDeviceCPU;

  float* input1_data = (float*)malloc(m * sizeof(float));
  float* input2_data = (float*)malloc(m * sizeof(float));

  float* output_data = (float*)malloc(m * sizeof(float));

  for (int i = 0; i < m; i++) {
    input1_data[i] = i;
    input2_data[i] = 0;
  }

  tensor::Tensor input1(dtype, m, false, nullptr, input1_data);
  tensor::Tensor input2(dtype, m, false, nullptr, input2_data);
  tensor::Tensor output(dtype, m, false, nullptr, output_data);

  kernel::get_add_kernel(device_type)(input1, input2, output, nullptr);
  for (int32_t i = 0; i < 10; i++) {
    std::cout << "\t\toutput" << "[" << i << "]" << "=" << output_data[i] << std::endl;
  }
}

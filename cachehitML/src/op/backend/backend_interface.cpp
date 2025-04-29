#include <backend/backend_interface.h>
#include <base/base.h>

/// cuda backend
#include "cuda-backend/add_kernel.cuh"
#include "cuda-backend/emb_kernel.cuh"
#include "cuda-backend/matmul_kernel.cuh"
// #include "cuda-backend/mha_kernel.cuh"
// #include "cuda-backend/rmsnorm_kernel.cuh"
// #include "cuda-backend/rope_kernel.cuh"
// #include "cuda-backend/scale_kernel.cuh"
// #include "cuda-backend/scale_sum_kernel.cuh"
// #include "cuda-backend/softmax_kernel.cuh"
// #include "cuda-backend/swiglu_kernel.cuh"
/// x86cpu backend
#include "x86cpu-backend/add_kernel.h"
#include "x86cpu-backend/emb_kernel.h"
#include "x86cpu-backend/matmul_kernel.h"
// #include "x86cpu-backend/mha_kernel.h"
// #include "x86cpu-backend/rmsnorm_kernel.h"
// #include "x86cpu-backend/rope_kernel.h"
// #include "x86cpu-backend/scale_kernel.h"
// #include "x86cpu-backend/scale_sum_kernel.h"
// #include "x86cpu-backend/softmax_kernel.h"
// #include "x86cpu-backend/swiglu_kernel.h"

namespace kernel {
namespace ML = cachehitML;
AddKernel get_add_kernel(ML::DeviceType device_type) {
  if (device_type == ML::DeviceType::kDeviceCPU) {
    return add_kernel_cpu;
  } /*else if (device_type == ML::DeviceType::kDeviceCUDA) {
    return add_kernel_cuda;
  } */
  else {
    LOG(FATAL) << "Unknown device type for add kernel.";
    return nullptr;
  }
}

EmbeddingKernel get_emb_kernel(ML::DeviceType device_type) {
  if (device_type == ML::DeviceType::kDeviceCPU) {
    return emb_kernel_cpu;
  } /*else if (device_type == ML::DeviceType::kDeviceCUDA) {
    return emb_kernel_cuda;
  } */
  else {
    LOG(FATAL) << "Unknown device type for emb kernel.";
    return nullptr;
  }
}

MatmulKernel get_matmul_kernel(ML::DeviceType device_type) {
  if (device_type == ML::DeviceType::kDeviceCPU) {
    return matmul_kernel_cpu;
  } else if (device_type == ML::DeviceType::kDeviceCUDA) {
    return matmul_kernel_cuda;
  } else {
    LOG(FATAL) << "Unknown device type for matmul kernel.";
    return nullptr;
  }
}

// MatmulKernelQuant get_matmul_quant_kernel(ML::DeviceType device_type) {
//   if (device_type == ML::DeviceType::kDeviceCUDA) {
//     return matmul_quant_kernel_cuda;
//   } else {
//     LOG(FATAL) << "UnSupport device type for matmul quant kernel.";
//     return nullptr;
//   }
// }

// MHAKernel get_mha_kernel(ML::DeviceType device_type) {
//   if (device_type == ML::DeviceType::kDeviceCPU) {
//     return mha_kernel_cpu;
//   } else if (device_type == ML::DeviceType::kDeviceCUDA) {
//     return mha_kernel_cuda;
//   } else {
//     LOG(FATAL) << "Unknown device type for mha kernel.";
//     return nullptr;
//   }
// }

// RoPEKernel get_rope_kernel(ML::DeviceType device_type) {
//   if (device_type == ML::DeviceType::kDeviceCPU) {
//     return rope_kernel_cpu;
//   } else if (device_type == ML::DeviceType::kDeviceCUDA) {
//     return rope_kernel_cuda;
//   } else {
//     LOG(FATAL) << "Unknown device type for rope kernel.";
//     return nullptr;
//   }
// }

// ScaleKernel get_scale_kernel(ML::DeviceType device_type) {
//   if (device_type == ML::DeviceType::kDeviceCPU) {
//     return scale_kernel_cpu;
//   } else {
//     LOG(FATAL) << "Unknown device type for scale kernel.";
//     return nullptr;
//   }
// }

}  // namespace kernel
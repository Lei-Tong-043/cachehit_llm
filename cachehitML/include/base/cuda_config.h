#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace op {
struct CudaConfig {
  cudaStream_t stream = nullptr;
  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
};
}  // namespace op
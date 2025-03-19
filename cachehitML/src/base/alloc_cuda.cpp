#include <cuda_runtime_api.h>

#include "base/alloc.h"

namespace cachehitML {

CUDAAllocator::CUDAAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

// creat a big buffer from cuda mem pool
void* CUDAAllocator::ML_malloc(size_t byte_size) const {
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  CHECK_EQ(state, cudaSuccess);
  // big_buffer == mem > 1MB
  if (byte_size > 1024 * 1024) {
    auto& big_buffers = big_buffers_map_[id];
    int sel_id = -1;
    for (int i = 0; i < big_buffers.size(); i++) {
      // 1.big_buffer is not busy
      // 2.big_buffer is big enough
      // 3.big_buffer is not too small
      if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
          big_buffers[i].byte_size - byte_size < 1024 * 1024) {
        // not find buffer
        // or
        // need buffer is smaller than current buffer
        // so choose current buffer
        if (sel_id == -1 ||
            big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
          // find a buffer and do not search again
          sel_id = i;
        }
      }
    }
    // find buffer from mem pool, use it and return
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
    }
    // if do not find used buffer
    // create new buffer
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (state != cudaSuccess) {
      char buf[256];
      snprintf(buf, 256, "Error: CUDA error when allocating %lu MB mem!",
               cudaGetErrorString(state), byte_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
  }

  // small buffer part
  // mem <= 1MB
  auto& cuda_buffers = cuda_buffers_map_[id];
  for (int i = 0; i < cuda_buffers.size(); i++) {
    // 1.cuda buffer big enough
    // 2.cuda buffer is not busy
    if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
      no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
      cuda_buffers[i].busy = true;
      return cuda_buffers[i].data;
    }
  }
  void* ptr = nullptr;
  state = cudaMalloc(&ptr, byte_size);
  if (state != cudaSuccess) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB memory! maybe there's "
             "no enough memory "
             "left on  device.",
             byte_size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  cuda_buffers.emplace_back(ptr, byte_size, true);
  return ptr;
}

// release memory
//
void CUDAAllocator::ML_release(void* ptr) const {
  // if ptr is nullptr or cuda
  if (!ptr) {
    return;
  }
  if (cuda_buffers_map_.empty()) {
    return;
  }
  cudaError_t state = cudaSuccess;
  for (auto it : cuda_buffers_map_) {
    if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
      auto& cuda_buffers = it.second;
      std::vector<CudaMemoryBuffer> temp;
      for (int i = 0; i < cuda_buffers.size(); i++) {
        if (!cuda_buffers[i].busy) {
          state = cudaSetDevice(it.first);
          state = cudaFree(cuda_buffers[i].data);
          CHECK(state == cudaSuccess)
              << "Error: CUDA error when releasing memory." << it.first;
        } else {
          temp.emplace_back(cuda_buffers[i]);
        }
      }
      cuda_buffers.clear();
      it.second = temp;
      no_busy_cnt_[it.first] = 0;
    }
  }
  // first  = map_id
  // second = buffer[map_id]
  for (auto& it : cuda_buffers_map_) {
    auto& cuda_buffers = it.second;
    for (int i = 0; i < cuda_buffers.size(); i++) {
      if (cuda_buffers[i].data == ptr) {
        no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
        cuda_buffers[i].busy = false;
        return;
      }
    }
    auto& big_buffers = big_buffers_map_[it.first];
    for (int i = 0; i< big_buffers.size(); i++) {
      if(big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        return;
      }
    }
  }
  state = cudaFree(ptr);  
  CHECK(state == cudaSuccess) << "Error: CUDA error when releasing memory.";
}

std::shared_ptr<CUDAAllocator> CUDAAllocatorFactory::instance_ = nullptr;

}  // namespace cachehitML
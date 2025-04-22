#include "tensor/tensor.h"

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <numeric>

namespace tensor {
namespace ML = cachehitML;

template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
  if (begin >= end) {
    return 0;
  }
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}

static size_t data_type_size(ML::DataType data_type) {
  switch (data_type) {
    case ML::DataType::kFP32:
    case ML::DataType::kInt32:
      return 4; 
    case ML::DataType::kInt8:
      return 1;
    default:
      LOG(FATAL) << "Unsupported data type";
      return 0;
  }
}

Tensor::Tensor(ML::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<ML::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  size_ = dim0;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    if (ptr != nullptr) {
      CHECK(need_alloc == false)
          << "The need_alloc is true when ptr parameter is not a null pointer.";
      init_buffer(alloc, data_type_, need_alloc, ptr);
    }
  }
}

// Tensor::Tensor(ML::DataType data_type, int32_t dim0, int32_t dim1,
//                bool need_alloc, std::shared_ptr<ML::DeviceAllocator> alloc,
//                void* ptr)
//     : data_type_(data_type) {
//   dims.push_back(dim0);
//   dims.push_back(dim1);
//   size_ = dim0 * dim1;
//   if (need_alloc && alloc) {
//     allocate(alloc);
//   } else {
//     if (ptr != nullptr) {
//       CHECK(need_alloc == false)
//           << "The need_alloc is true when ptr parameter is not a null pointer.";
//       init_buffer(alloc, data_type_, need_alloc, ptr);
//     }
//   }
// }

// Tensor::Tensor(ML::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
//                bool need_alloc, std::shared_ptr<ML::DeviceAllocator> alloc,
//                void* ptr)
//     : data_type_(data_type) {
//   dims_.push_back(dim0);
//   dims_.push_back(dim1);
//   dims_.push_back(dim2);
//   size_ = dim0 * dim1 * dim2;
//   if (need_alloc && alloc) {
//     allocate(alloc);
//   } else {
//     init_buffer(alloc, data_type_, need_alloc, ptr);
//   }
// }

// Tensor::Tensor(ML::DataType data_type, std::vector<int32_t> dims,
//                bool need_alloc, std::shared_ptr<ML::DeviceAllocator> alloc,
//                void* ptr)
//     : dims_(std::move(dims)), data_type_(data_type) {
//   size_ = reduce_dimension(dims_.begin(), dim_.end(), 1);
//   if (need_alloc && alloc) {
//     allocate(alloc);
//   } else {
//     init_buffer(alloc, data_type_, need_alloc, ptr);
//   }
// }

// void Tensor::to_cuda(cudaStream_t stream) {
//   CHECK_NE(buffer_, nullptr);
//   const ML::DeviceType device_type = buffer_->device_type();
//   if (device_type == ML::DeviceType::kDeviceUnknown) {
//     LOG(ERROR) << "The device type of the tensor is unknown!";
//   } else if (device_type == ML::DeviceType::kDeviceCPU) {
//     size_t byte_size = this->byte_size();
//     auto cu_alloc = ML::CUDAAllocatorFactory::get_instance();
//     auto cu_buffer = std::make_shared<ML::Buffer>(byte_size, cu_alloc);
//     cu_alloc->ML_memcpy(cu_buffer->ptr(), buffer_->ptr(), byte_size,
//                         ML::MemcpyKind::kMemcpyCPU2CUDA, stream);
//     this->buffer_ = cu_buffer;
//   } else {
//     LOG(INFO) << "The device type of the tensor is already cuda.";
//   }
// }

// void Tensor::to_cpu() {
//   CHECK_NE(buffer_, nullptr);
//   const ML::DeviceType device_type = this->device_type();

//   if (device_type == ML::DeviceType::kDeviceUnknown) {
//     LOG(ERROR) << "The device type of the tensor is unknown!";
//   } else if (device_type == ML::DeviceType::kDeviceCUDA) {
//     size_t byte_size = this->byte_size();
//     auto cpu_alloc = ML::CPUAllocatorFactory::get_instance();
//     auto cpu_buffer = std::shared_ptr<ML::Buffer>(byte_size, cpu_alloc);
//     this->buffer_ = cpu_buffer;
//   } else {
//     LOG(INFO) << "The device type of the tensor is already cpu.";
//   }
// }

size_t Tensor::size() const { return this->size_; }

// int32_t Tensor::get_dim(int32_t idx) const {
//   CHECK_GE(idx, 0);
//   CHECK_LT(idx, this->dims_.size());
//   return this->dims_.at(idx);
// }

// ML::DeviceType Tensor::device_type() const {
//   if (!buffer_) {
//     return ML::DeviceType::kDeviceUnknwon;
//   }
//   return buffer_->device_type();
// }

// bool Tensor::assign(std::shared_ptr<ML::Buffer> buffer) {
//   if (!buffer) {
//     LOG(ERROR) << "The buffer parameter in the assign funtion is null pointer!";
//     return false;
//   }
//   if (buffer_) {
//     if (buffer_->device_type() != buffer->device_type()) {
//       LOG(ERROR) << "The device type of the new buffer is different from the "
//                     "original one.1";
//     }
//   }

//   size_t byte_size = this->byte_size();
//   if (byte_size > buffer->byte_size()) {
//     LOG(ERROR) << "The size of buffer is too small for the tensor!";
//     return false;
//   }
//   buffer_ = buffer;
//   return true;
// }

bool Tensor::allocate(std::shared_ptr<ML::DeviceAllocator> allocator, bool need_alloc) {
  if (!allocator) {
    LOG(ERROR) << "The allocator parameter in the allocate funtion is nullptr!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (!byte_size) {
    LOG(ERROR) << "The byte_size parameter in the allocate funtion is equal to 0!";
    return false;
  }

  if (buffer_ && byte_size <= buffer_->byte_size()) {
    if (!need_alloc) {
      return true;
    }
  }

  buffer_ = std::make_shared<cachehitML::Buffer>(byte_size, allocator, nullptr);
  if (!buffer_->ptr()) {
    LOG(ERROR) << "The memory allocated is a null pointer!";
    return false;
  }
  return true;
}

// const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }

// void Tensor::set_device_type(ML::DeviceType device_type) const {
//   if (buffer_) {
//     buffer_->set_device_type(device_type);
//   }
// }

// void Tensor::reset(ML::DataType data_type, const std::vector<int32_t>& dims) {
//   this->data_type_ = data_type;
//   this->dims_ = dims;
//   this->size_ = reduce_dimension(dims.begin(), dims.end(), 1);
//   this->buffer_ = nullptr;
// }

// int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

// ML::DataType Tensor::data_type() const { return data_type_; }

// void Tensor::reshape(const std::vector<int32_t>& dims) {
//   size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
//   if (!buffer_) {
//     this->dims_ = dims;
//     this->size_ = size;
//     return;
//   }

//   if (size > size_) {
//     auto new_buffer = std::make_shared<ML::Buffer>(
//         size * ML::DataTypeSize(this->data_type_), buffer->allocator());
//     CHECK(new_buffer->allocate());
//     new_buffer_->copy_from(buffer_.get());
//     this->buffer_ = new_buffer;
//   }
//   this->dims_ =dims;
//   this->size_ =size;
// }

// std::shared_ptr<ML::Buffer> Tensor::get_buffer() const { return buffer_;}

// Tensor Tensor::clone(){
//   Tensor new_tensor = *this;
//   size_t byte_size = this->byte_size();

//   auto allocator =  buffer_->allocatorA();
//   new_tensor.buffer_ = std::make_shared<ML::Buffer>(byte_size, allocator);
//   new_tensor.buffer_->copy_from(buffer.get());
//   return new_tensor;
// }

size_t Tensor::byte_size() const { return this->size() * DataTypeSize(data_type_); }

// std::vector<size_t> Tensor::strides() const{
//   std::vector<size_t> strides;
//   if(!dims.empty()){
//     for(int32_t i =0; i<dims_.size() -1 ;++i){
//       size_t stride = reduce_dimension(dims_.begin() + i +1, dims.edm(),1);
//       strides.push_back(stride);
//     }
//     strides.push_back(1);
//   }
//   return strides;
// }

// bool Tensor::is_empty() const{
//   return size_ == 0|| buffer_ ==nullptr|| buffer_->ptr() == nullptr;
// }

void Tensor::init_buffer(std::shared_ptr<ML::DeviceAllocator> alloc, ML::DataType data_type,
                         bool need_alloc, void* p) {
  if (!need_alloc && !alloc) {
    std::shared_ptr<ML::Buffer> buffer =
        std::make_shared<ML::Buffer>(data_type_size(data_type) * size_, nullptr, p, true);
    this->buffer_ = buffer;
  } else {
    allocate(alloc, true);
  }
}

}  // namespace tensor
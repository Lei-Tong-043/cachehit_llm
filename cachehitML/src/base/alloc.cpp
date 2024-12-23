#include <base/alloc.h>
#include <cuda_runtime.h>

namespace cachehitML{
// all kind memcpy
void DeviceAllocator::MLmemcpy(void* dest_ptr, const void* src_ptr, size_t byte_size, 
                            MemcpyKind memcpy_kind, void* stream, bool need_syncs) const{
    // check arg vaild
    CHECK_NE(src_pet, nullptr);
    CHECK_NE(dest_ptr, nullptr);    
    if (!byte_size){
        return ;
    }
    //memcpy logic
    cudastream_t stream_ = nullptr;
    if (stream){
        stream_ =static_cast<CUstream_st*>(stream);
    }
    if (memcpy_kind == MemcpyKind::MemcpyCPU2CPU){
        std::memcpu(dest_ptr, src_ptr, byte_size);
    }else if(memcpy_kind == MemcpyKind::MemcpyCPU2CUDA){
        if(!stream_){
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
        }else{
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
        }
    }else if (memcpy_kind == MemcpyKind::MemcpyCUDA2CUDA){
        if(!stream_){
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
        }else{
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
        }
    }else if (memcpy_kind == MemcpyKind::MemcpyCUDA2CUDA){
        if(!stream_){
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
        }else{
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
        }
    }else if (memcpy_kind == MemcpyKind::MemcpyCUDA2CUDA){
        if(!stream_){
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
        }else{
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
        }
    }else{
        LOG(FATAL) << "Unknown memcpy kind: "<< int(memcpy_kind);
    }
    if(need_sync){
        cudaDeviceSynchronize();
    }
}

// all kind memset zero
void DeviceAllocator::MLmemset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync){
    CHECK(device_type_ != DeviceType::DEVICE_UNKNOWN);
    if(device_type_ == DeviceType::DEVICE_X86CPU){
        std::memset(ptr, 0, byte_size);
    }else {
        if (stream){
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            cudaMemsetAsync(ptr, 0, byte_size, stream_);
        }else{
            cudaMemset(ptr, 0, byte_size);
        }
        if(need_sync){
            cudaDeviceSynchronize();
        }
    }
}

} // namespace cachehitML 
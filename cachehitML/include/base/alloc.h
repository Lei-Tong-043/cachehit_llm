#ifndef CACHEHIT_INCLUDE_BASE_ALLOC_H_
#define CACHEHIT_INCLUDE_BASE_ALLOC_H_

#include "base.h"
#include <map>
#include <memory>

namespace cachehitML {

enum class MemcpyKind{
    MemcpyCPU2CPU,
    MemcpyCPU2CUDA,
    MemcpyCUDA2CPU,
    MemcpyCUDA2CUDA,
};
// base class 
class DeviceAllocator{
public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

    virtual DeviceType device_type() const {
        return device_type_;
    }
    
    virtual void MLrelease(void* ptr) const = 0;

    virtual void* MLalloc(size_t byte_size) const =0;

    virtual void MLmemcpy(void* dest_ptr, const void* src_ptr, size_t byte_size,
            MemcpyKind memcpy_kind = MemcpyKind::MemcpyCPU2CPU, void* stream = nullptr,
            bool need_sync = false) const;

    virtual void MLmemset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync=false);

private:
    DeviceType device_type_ = DeviceType::DEVICE_UNKNOWN;
};

struct CudaMemBuffer
{   
    size_t byte_size_;
    bool busy_;
    void *data_;

    CudaMemBuffer() = default;

    CudaMemBuffer(void* data, size_t byte_size, bool busy) 
                : data_(data), byte_size_(byte_size), busy_(busy){};
};

class CPUAlloctor : public DeviceAllocator {
public:
    explicit CPUAlloctor();

    void* MLalloc(size_t byte_size) const override;

    void MLrelease(void* ptr) const override;
};


class CUDAAllocator : public DeviceAllocator{
public:
    explicit CUDAAllocator();

    void *MLalloc(size_t byte_size) const override;

    void MLrelease(void * ptr) const override;

private:
    mutable std::map<int, size_t> no_busy_cnt_;
    mutable std::map<int, std::vector<CudaMemBuffer>> big_buffers_map_;
    mutable std::map<int, std::vector<CudaMemBuffer>> cuda_buffers_map_;
};

class CPUAllocFactory{
public:
    static std::shared_ptr<CPUAlloctor> get_instance() {
        if( instance == nullptr){
            instance = std::make_shared<CPUAlloctor>();
        }
        return instance;
    }
private:
    static std::shared_ptr<CPUAlloctor> instance;
};

class CUDAAllocFactory {
public:
    static std::shared_ptr<CUDAAllocator> get_instance(){
        if(instance == nullptr){
            instance = std::make_shared<CUDAAllocator>();
        }
        return instance;
    }
private:
    static std::shared_ptr<CUDAAllocator> instance;
};

class DeviceAllocFactory{
public:
    static std::shared_ptr<DeviceAllocator> get_instance(cachehitML::DeviceType device_type){
        if(device_type == cachehitML::DeviceType::DEVICE_CPU){
            return CPUAllocFactory::get_instance();
        }else if(device_type == cachehitML::DeviceType::DEVICE_NVGPU){
            return CUDAAllocFactory::get_instance();
        }else{
            LOG(FATAL) << "This device type of allocator is not supported!";
            return nullptr;
        }
    }
};

}
#endif
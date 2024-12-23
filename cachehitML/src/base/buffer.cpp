#include <base/buffer.h>
#include <glog/logging.h>

namespace base{

Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator,
    void* ptr, bool use_external)
:   byte_size_(byte_size),
    allocator_(allocator),
    ptr_(ptr),
    use_external_(use_external_){
    
    if(!ptr_ && allocator_){
        device_type = allocator_->device_type();
        use_external_ = false;
        ptr_ = allocator->allocate(byte_size);
    }
}
Buffer::~Buffer(){
    if (!use_external_){
        if(ptr_ && allocator_){
            allocator_->release(ptr_);
            ptr_ = nullptr;
        }
    }
}

void* Buffer::ptr(){
    return ptr_;
}

const void* Buffer::ptr() const{
    return ptr_;
}

size_t Buffer::byte_size() const{
    return byte_size_;
}

bool Buffer::allocate() {
    if (allocator_ && byte_size_ != 0){
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size_);
        if(!ptr_){
            return false;
        }else{
            return true;
        }
    }else{
        return false;
    }
}

std::shared_ptr<DeviceAllocator> Buffer::allocator() const{
    return allocator_;
}

void Buffer::copy_from(const Buffer& buffer) const  {
    CHECK(allocator_ != nullptr);
    CHECK(buffer.ptr != nullptr);

    size_t byte_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
    const DeviceType& buffer_device = buffer.device_type();
    const DeviceType& current_device = this->device_type();
    CHECK(buffer_device != DeviceType::DEVICE_UNKNOWN &&
        current_device != DeviceType::DEVICE_UNKNOWN);

    if(buffer_device == DeviceType::DEVICE_X86CPU && 
        current_device == DeviceType::DEVICE_X86CPU){
        return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size);
    }else if (buffer_device == DeviceType::DEVICE_NVGPU &&
        current_device == DeviceType::DEVICE_X86CPU){
        return allocator_->mem
    }
}


}
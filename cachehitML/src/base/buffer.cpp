// #include "base/buffer.h"
// #include <glog/logging.h>

// namespace cachehitML{

// Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator,void* ptr, bool use_external)
//   : byte_size_(byte_size),
//     allocator_(allocator),
//     ptr_(ptr),
//     use_external_(use_external){
    
//     if(!ptr_ && allocator_){
//         device_type_ = allocator_->device_type();
//         use_external_ = false;
//         ptr_ = allocator_->MLalloc(byte_size);
//     }
// }
// Buffer::~Buffer(){
//     if (!use_external_){
//         if(ptr_ && allocator_){
//             allocator_->MLrelease(ptr_);
//             ptr_ = nullptr;
//         }
//     }
// }

// void* Buffer::ptr(){
//     return ptr_;
// }

// const void* Buffer::ptr() const{
//     return ptr_;
// }

// size_t Buffer::byte_size() const{
//     return byte_size_;
// }

// bool Buffer::MLalloc() {
//     if (allocator_ && byte_size_ != 0){
//         use_external_ = false;
//         ptr_ = allocator_->MLalloc(byte_size_);
//         if(!ptr_){
//             return false;
//         }else{
//             return true;
//         }
//     }else{
//         return false;
//     }
// }

// std::shared_ptr<DeviceAllocator> Buffer::allocator() const{
//     return allocator_;
// }

// void Buffer::copy_from(const Buffer& buffer) const  {
//     CHECK(allocator_ != nullptr);
//     CHECK(buffer.ptr() != nullptr);

//     size_t byte_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
//     const DeviceType& buffer_device = buffer.device_type();
//     const DeviceType& current_device = this->device_type();
//     CHECK(buffer_device != DeviceType::DEVICE_UNKNOWN && current_device != DeviceType::DEVICE_UNKNOWN);

//     // cpu to cpu
//     if(buffer_device == DeviceType::DEVICE_CPU && current_device == DeviceType::DEVICE_CPU){
//         return allocator_->MLmemcpy(this->ptr_, buffer.ptr(), byte_size);
//     // cpu to nvgpu 
//     }else if (buffer_device == DeviceType::DEVICE_NVGPU && current_device == DeviceType::DEVICE_CPU){
//         return allocator_->MLmemcpy(this->ptr_, buffer.ptr(), byte_size, MemcpyKind::MemcpyCPU2CUDA);
//     // nvgpu to cpu
//     }else if (buffer_device == DeviceType::DEVICE_CPU && current_device == DeviceType::DEVICE_NVGPU ){
//         return allocator_->MLmemcpy(this->ptr_, buffer.ptr(), byte_size, MemcpyKind::MemcpyCUDA2CPU);
//     // nvgpu to nvgpu
//     }else{
//         return allocator_->MLmemcpy(this->ptr_, buffer.ptr(), byte_size, MemcpyKind::MemcpyCUDA2CUDA);
//     }
// }

// DeviceType Buffer::device_type() const{
//     return device_type_;
// }

// void Buffer::set_device_type(DeviceType device_type){
//     device_type_ = device_type;
// }

// std::shared_ptr<Buffer> Buffer::get_shared_from_this(){
//     return shared_from_this();
// }

// bool Buffer::is_external() const{
//     return this->use_external_;
// }

// }   // namespace cachehitML
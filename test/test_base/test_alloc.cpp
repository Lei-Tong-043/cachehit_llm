#include <gtest/gtest.h>
#include <base/alloc.h>
#include <string>

using namespace cachehitML;

class AllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_allocator = CPUAllocFactory::get_instance();
        cuda_allocator = CUDAAllocFactory::get_instance();
    }

    std::shared_ptr<CPUAlloctor> cpu_allocator;
    std::shared_ptr<CUDAAllocator> cuda_allocator;
};

TEST_F(AllocatorTest, CPUAllocationTest) {
    size_t size = 1024;
    void* ptr = cpu_allocator->MLalloc(size);
    ASSERT_NE(ptr, nullptr);  // Ensure allocation is successful.

    cpu_allocator->MLrelease(ptr);
}

TEST_F(AllocatorTest, CUDAAllocationTest) {
    size_t size = 1024;
    void* ptr = cuda_allocator->MLalloc(size);
    ASSERT_NE(ptr, nullptr);  // Ensure allocation is successful.

    cuda_allocator->MLrelease(ptr);
}

TEST_F(AllocatorTest, DeviceAllocFactoryTest) {
    auto cpu_alloc = DeviceAllocFactory::get_instance(DeviceType::DEVICE_CPU);
    ASSERT_NE(cpu_alloc, nullptr);
    EXPECT_EQ(cpu_alloc->device_type(), DeviceType::DEVICE_CPU);

    auto cuda_alloc = DeviceAllocFactory::get_instance(DeviceType::DEVICE_NVGPU);
    ASSERT_NE(cuda_alloc, nullptr);
    EXPECT_EQ(cuda_alloc->device_type(), DeviceType::DEVICE_NVGPU);
}

TEST_F(AllocatorTest, MemcpyTest) {
    size_t size = 1024;
    void* src = cpu_allocator->MLalloc(size);
    void* dest = cpu_allocator->MLalloc(size);

    memset(src, 42, size);  // Fill source with a known value.
    cpu_allocator->MLmemcpy(dest, src, size);
    ASSERT_EQ(memcmp(src, dest, size), 0);  // Ensure memory contents match.

    cpu_allocator->MLrelease(src);
    cpu_allocator->MLrelease(dest);
}

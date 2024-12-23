#include <gtest/gtest.h>
#include <base/alloc.h>
#include <string>

using namespace cachehitML;

//test DeviceAllocator::MLcopy
TEST(DeviceAllocator, MLMemcpy){
    float *cpu;
    float *gpu;
    DeviceAlloctor::MLalloc
}
#include <base/buffer.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace cachehitML;

TEST(BufferTest, DefaultBuffer) {
  Buffer test_buffer;
  LOG(INFO) << "test";
  ASSERT_EQ(test_buffer.byte_size(), 0);
  ASSERT_EQ(test_buffer.device_type(), DeviceType::kDeviceUnknown);
}


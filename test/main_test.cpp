#include "glog/logging.h"
#include "gtest/gtest.h"

int main(int argc, char* argv[]) {
  // 初始化 Google Logging
  google::InitGoogleLogging(argv[0]);  // 使用 argv[0] 作为程序名

  // 初始化 Google Test 框架
  // Google Test 会解析 argc 和 argv 中的测试相关命令行参数
  ::testing::InitGoogleTest(&argc, argv);

  FLAGS_logtostderr = true;
  // 运行所有注册的测试用例
  int result = RUN_ALL_TESTS();  // 通常会返回测试结果状态

  // 关闭 Google Logging (可选，但推荐在程序结束前调用)
  google::ShutdownGoogleLogging();

  return result;  // 返回测试结果
}

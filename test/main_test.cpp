#include "gtest/gtest.h" // 包含 Google Test 头文件
#include "glog/logging.h" // 包含 Google Logging 头文件

// 你项目中其他可能需要在 main 中初始化的内容，通常不多

int main(int argc, char* argv[]) {
    // --- Google Logging 初始化 ---
    // 初始化 Google Logging 库。
    // argv[0] 是程序的名称，glog 会用它来生成日志文件名。
    google::InitGoogleLogging(argv[0]);

    // 设置 Google Logging 的 FLAGS。
    // 推荐在单元测试中将日志输出到标准错误，方便查看。
    // 这行代码会使所有日志（取决于FLAGS_minloglevel）同时输出到终端的标准错误流。
    FLAGS_logtostderr = true;

    // 可选：如果你想控制输出到 stderr 的最低日志级别，可以设置 FLAGS_stderrthreshold。
    // 例如，设置 FLAGS_stderrthreshold = 0; 会将 INFO 及以上级别的日志输出到 stderr。
    // FLAGS_stderrthreshold = 0; // 取消注释即可启用

    // 可选：如果你想禁用某些日志级别（例如，不显示 INFO 级别的日志），可以设置 FLAGS_minloglevel。
    // FLAGS_minloglevel = google::WARNING; // 只显示 WARNING, ERROR, FATAL

    // 可选：设置日志文件输出目录，如果需要将日志写入文件。
    // FLAGS_log_dir = "/path/to/your/log/directory"; // 取消注释并修改路径即可启用文件输出

    // --- Google Test 初始化 ---
    // 初始化 Google Test 框架。
    // 它会解析 argc 和 argv 中的 Google Test 命令行参数。
    ::testing::InitGoogleTest(&argc, argv);

    // --- 运行所有测试 ---
    // 运行所有用 TEST() 或 TEST_F() 注册的测试用例。
    // 返回值为 0 表示所有测试通过，非 0 表示有测试失败。
    int result = RUN_ALL_TESTS();

    // --- Google Logging 关闭 (可选) ---
    // 关闭 Google Logging 库。
    // 在 main 函数的末尾调用是可选的，程序退出时会自动关闭。
    // google::ShutdownGoogleLogging(); // 通常可以注释掉

    // 返回测试结果
    return result;
}
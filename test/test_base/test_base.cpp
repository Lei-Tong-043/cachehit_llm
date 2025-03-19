#include <gtest/gtest.h>
#include <base/base.h>
#include <string>

// 使用namespace来简化代码
using namespace cachehitML;

// 测试 Status 类
TEST(StatusTest, DefaultConstructor) {
    // 默认构造函数应该生成成功状态
    Status status;
    EXPECT_EQ(status.get_err_code(), StatusCode::kSuccess);
    EXPECT_EQ(status.get_err_msg(), "");
}

TEST(StatusTest, ParametrizedConstructor) {
    // 测试带有参数的构造函数
    Status status(StatusCode::kModelParseError, "Model parsing failed");
    EXPECT_EQ(status.get_err_code(), StatusCode::kModelParseError);
    EXPECT_EQ(status.get_err_msg(), "Model parsing failed");
}

TEST(StatusTest, AssignmentOperator) {
    // 测试赋值操作符
    Status status1(StatusCode::kInternalError, "Internal server error");
    Status status2 = status1;  // 调用拷贝构造
    EXPECT_EQ(status2.get_err_code(), StatusCode::kInternalError);
    EXPECT_EQ(status2.get_err_msg(), "Internal server error");
}

TEST(StatusTest, ComparisonOperator) {
    // 测试相等与不等操作符
    Status status1(StatusCode::kSuccess, "");
    Status status2(StatusCode::kModelParseError, "Parsing error");
    
    EXPECT_TRUE(status1 == StatusCode::kSuccess);
    EXPECT_FALSE(status2 == StatusCode::kSuccess);
    EXPECT_TRUE(status2 != StatusCode::kSuccess);
}

TEST(StatusTest, BooleanOperator) {
    // 测试布尔类型转换
    Status successStatus(StatusCode::kSuccess, "Everything is fine");
    Status errorStatus(StatusCode::kModelParseError, "Model error");

    EXPECT_TRUE(successStatus);
    EXPECT_FALSE(errorStatus);
}

// 测试 CHECK_ERR 宏
TEST(CheckErrTest, SuccessCase) {
    // 在此使用 CHECK_ERR 宏直接测试
    EXPECT_NO_THROW(STATUS_CHECK(error::Success()));  // 测试成功的情况
}
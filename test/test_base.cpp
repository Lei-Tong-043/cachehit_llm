#include <gtest/gtest.h>
#include <base/base.h>
#include <string>

// 使用namespace来简化代码
using namespace cachehitML;

// 测试 Status 类
TEST(StatusTest, DefaultConstructor) {
    // 默认构造函数应该生成成功状态
    Status status;
    EXPECT_EQ(status.get_err_code(), StatusCode::SUCCESS);
    EXPECT_EQ(status.get_err_msg(), "");
}

TEST(StatusTest, ParametrizedConstructor) {
    // 测试带有参数的构造函数
    Status status(StatusCode::MODEL_PARSE_ERROR, "Model parsing failed");
    EXPECT_EQ(status.get_err_code(), StatusCode::MODEL_PARSE_ERROR);
    EXPECT_EQ(status.get_err_msg(), "Model parsing failed");
}

TEST(StatusTest, AssignmentOperator) {
    // 测试赋值操作符
    Status status1(StatusCode::INTERNAL_ERROR, "Internal server error");
    Status status2 = status1;  // 调用拷贝构造
    EXPECT_EQ(status2.get_err_code(), StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(status2.get_err_msg(), "Internal server error");
}

TEST(StatusTest, ComparisonOperator) {
    // 测试相等与不等操作符
    Status status1(StatusCode::SUCCESS, "");
    Status status2(StatusCode::MODEL_PARSE_ERROR, "Parsing error");
    
    EXPECT_TRUE(status1 == StatusCode::SUCCESS);
    EXPECT_FALSE(status2 == StatusCode::SUCCESS);
    EXPECT_TRUE(status2 != StatusCode::SUCCESS);
}

TEST(StatusTest, BooleanOperator) {
    // 测试布尔类型转换
    Status successStatus(StatusCode::SUCCESS, "Everything is fine");
    Status errorStatus(StatusCode::MODEL_PARSE_ERROR, "Model error");

    EXPECT_TRUE(successStatus);
    EXPECT_FALSE(errorStatus);
}

// 测试 CHECK_ERR 宏
TEST(CheckErrTest, SuccessCase) {
    // 在此使用 CHECK_ERR 宏直接测试
    EXPECT_NO_THROW(CHECK_ML_ERR(error::Run_Success()));  // 测试成功的情况
}
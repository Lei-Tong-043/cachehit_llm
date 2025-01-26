#ifndef CACHEHIT_INCLUDE_BASE_BASE_H_
#define CACHEHIT_INCLUDE_BASE_BASE_H_
#include <glog/logging.h>
#include <cstdint>
#include <string>

// 显式标记未使用的变量或参数，避免编译器警告
#define UNUSED(expr) \
  do{                \
    (void)(expr);    \
  }while(0)

namespace model {
enum class ModelBufferType {
    kInputTokens = 0,
    kInputEmbeddings = 1,
    kOutputRMSNorm = 2,
    kKeyCache = 3,
    kValueCache = 4,
    kQuery = 5,
    kInputPos = 6,
    kScoreStorage = 7,
    kOutputMHA = 8,
    kAttnOutput = 9,
    kW1Output = 10,
    kW2Output = 11,
    kW3Output = 12,
    kFFNRMSNorm = 13,
    kForwardOutput = 15,
    kForwardOutputCPU = 16,
    kSinCache = 17,
    kCosCache = 18,
};
} // namespace model

namespace cachehitML {

} // namespace cachehitML

#endif

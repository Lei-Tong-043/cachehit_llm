#ifndef CACHEHIT_INCLUDE_BASE_BASE_H_
#define CACHEHIT_INCLUDE_BASE_BASE_H_
#include <glog/logging.h>

#include <cstdint>
#include <string>

// signed argument to avoid warning
#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

// model layer buffer type
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
}  // namespace model

namespace cachehitML {
// new device type
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
};

// new data type
enum class DataType : uint8_t {
  kUnkonwn = 0,
  kFP32 = 1,
  kInt8 = 2,
  kInt32 = 3,
};

// new model type
enum class ModelType : uint8_t {
  kModelUnknown = 0,
  kModelTypeLlama2 = 1,
};

// get data_type size (bytes)
inline size_t DataTypeSize(DataType data_type) {
  if (data_type == DataType::kFP32) {
    return sizeof(float);
  } else if (data_type == DataType::kInt8) {
    return sizeof(int8_t);
  } else if (data_type == DataType::kInt32) {
    return sizeof(int32_t);
  } else {
    return 0;
  }
}

// define nocopy class
class NoCopyable {
 protected:
  NoCopyable() = default;
  ~NoCopyable() = default;
  NoCopyable(const NoCopyable&) = delete;
  NoCopyable& operator=(const NoCopyable&) = delete;
};

// run model status code
enum StatusCode : uint8_t {
  kSuccess = 0,
  kFuntionUnImplement = 1,
  kPathNotValid = 2,
  kModelParseError = 3,
  kInternalError = 4,
  kKeyValueHasExist = 5,
  kInvalidArgument = 6,
};

enum class TokenizerType {
  kEncodeUnknown = -1,
  kEncodeSpe = 0,
  kEncodeBpe = 1,
};

// define status class to print error message
class Status {
 public:
  Status(int code = StatusCode::kSuccess, std::string err_msg = "");
  Status(const Status& other) = default;

  Status& operator=(const Status& other) = default;
  Status& operator=(int code);

  bool operator==(int code) const;
  bool operator!=(int code) const;

  ~Status() = default;

  operator int() const;
  operator bool() const;

  int32_t get_err_code() const;
  const std::string& get_err_msg() const;

  void set_err_msg(const std::string& err_msg);

 private:
  int code_ = StatusCode::kSuccess;
  std::string msg_;
};

// define error space to debug
namespace error {
#define STATUS_CHECK(call)                                                  \
  do {                                                                      \
    const cachehitML::Status& status = call;                                \
    if (!status) {                                                          \
      const size_t buf_size = 512;                                          \
      char buf[buf_size];                                                   \
      snprintf(                                                             \
          buf, buf_size - 1,                                                \
          "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", \
          __FILE__, __LINE__, int(status), status.get_err_msg().c_str());   \
      LOG(FATAL) << buf;                                                    \
    }                                                                       \
  } while (0)

Status Success(const std::string& err_msg = "");
Status FuntionUnImplement(const std::string& err_msg = "");
Status PathNotValid(const std::string& err_msg = "");
Status ModelParseError(const std::string& err_msg = "");
Status InternalError(const std::string& err_msg = "");
Status KeyValueHasExist(const std::string& err_msg = "");
Status InvalidArgument(const std::string& err_msg = "");
}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace cachehitML

#endif

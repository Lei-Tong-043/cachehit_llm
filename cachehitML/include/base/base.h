#ifndef CACHEHIT_INCLUDE_BASE_BASE_H_
#define CACHEHIT_INCLUDE_BASE_BASE_H_
#include <glog/logging.h>
#include <cstdint>
#include <string>

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

enum class DeviceType : uint8_t {
    DEVICE_UNKNOWN,
    DEVICE_X86CPU,
    DEVICE_NVGPU,
};

enum class DataType : uint8_t {
    DATA_TYPE_UNKNOWN,
    DATA_TYPE_FP32,
    DATA_TYPE_INT32,
    DATA_TYPE_INT8,
};

enum class ModelType : uint8_t {
    MODEL_UNKNOWN,
    MODEL_LLAMA2,
};

inline size_t DataTypeSize(DataType data_type) {
    switch(data_type) {
        case DataType::DATA_TYPE_FP32:
            return sizeof(float);
        case DataType::DATA_TYPE_INT32:
            return sizeof(int32_t);
        case DataType::DATA_TYPE_INT8:
            return sizeof(int8_t);
        default:
            LOG(ERROR) << "Unknown DataType!";
            return 0;
    }
}

class NoCopyable {
protected:
    NoCopyable() = default;
    ~NoCopyable() = default;
    NoCopyable(const NoCopyable&) = delete;
    NoCopyable& operator=(const NoCopyable&) = delete;
};

enum StatusCode : uint8_t {
    SUCCESS,
    FUNCTION_UNIMPL,
    PATH_NOT_VALID,
    MODEL_PARSE_ERROR,
    INTERNAL_ERROR,
    KEY_VALUE_HAS_EXIST,
    INVALID_ARGUMENT,
};

enum class TokenizerType {
    ENCODE_UNKNOWN,
    ENCODE_SPE,
    ENCODE_BPE,
};

class Status {
public:
    Status(int code = SUCCESS, std::string err_msg = "");
    Status(const Status& other) = default;
    Status& operator=(const Status& other) = default;
    Status& operator=(int code);
    bool operator==(int code) const;
    bool operator!=(int code) const;
    operator int() const;
    operator bool() const;
    int32_t get_err_code() const;
    const std::string& get_err_msg() const;
    void set_err_msg(const std::string& err_msg);

private:
    int code_ = SUCCESS;
    std::string msg_;
};

namespace error {
#define CHECK_ML_ERR(call) \
    do { \
        const cachehitML::Status& status = call; \
        if (!status) { \
            const size_t buf_size = 512; \
            char buf[buf_size]; \
            snprintf(buf, buf_size-1, \
                "Infer Error\n File:%s Line:%d\n Err_code:%d\n Err_msg:%s\n", \
                    __FILE__, __LINE__, int(status), status.get_err_msg().c_str()); \
            LOG(FATAL) << buf; \
        } \
    } while (0)

    Status Run_Success(const std::string& err_msg = "");
    Status Fuction_Unimpl(const std::string& err_msg = "");
    Status Path_Not_Valid(const std::string& err_msg = "");
    Status Model_Parse_Error(const std::string& err_msg = "");
    Status Internal_Error(const std::string& err_msg = "");
    Status Key_Value_Has_Exist(const std::string& err_msg = "");
    Status Invalid_Argument(const std::string& err_msg = "");

} // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x);

} // namespace cachehitML

#endif

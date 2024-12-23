#include "base/base.h"
#include <utility>

namespace cachehitML {

    Status::Status(int code, std::string err_msg): code_(code), msg_(std::move(err_msg)) {}

    Status& Status::operator=(int code) {
        code_ = code;
        return *this;
    }

    bool Status::operator==(int code) const {
        return code_ == code;
    }

    bool Status::operator!=(int code) const {
        return code_ != code;
    }

    Status::operator int() const {
        return code_;
    }

    Status::operator bool() const {
        return code_ == SUCCESS;
    }

    int32_t Status::get_err_code() const {
        return code_;
    }

    const std::string& Status::get_err_msg() const {
        return msg_;
    }

    void Status::set_err_msg(const std::string& err_msg) {
        msg_ = err_msg;
    }

namespace error {

    Status Run_Success(const std::string& err_msg) {
        return Status{SUCCESS, err_msg};
    }

    Status Fuction_Unimpl(const std::string& err_msg) {
        return Status{FUNCTION_UNIMPL, err_msg};
    }

    Status Path_Not_Valid(const std::string& err_msg) {
        return Status{PATH_NOT_VALID, err_msg};
    }

    Status Model_Parse_Error(const std::string& err_msg) {
        return Status{MODEL_PARSE_ERROR, err_msg};
    }

    Status Internal_Error(const std::string& err_msg) {
        return Status{INTERNAL_ERROR, err_msg};
    }

    Status Key_Value_Has_Exist(const std::string& err_msg) {
        return Status{KEY_VALUE_HAS_EXIST, err_msg};
    }

    Status Invalid_Argument(const std::string& err_msg) {
        return Status{INVALID_ARGUMENT, err_msg};
    }

} // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x) {
    os << x.get_err_msg();
    return os;
}

} // namespace cachehitML

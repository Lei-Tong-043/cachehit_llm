#include "base/base.h"
#include <string>

namespace cachehitML {

Status::Status(int code, std::string err_msg)
    : code_(code), msg_(std::move(err_msg)) {}

Status& Status::operator=(int code) {
  code_ = code;
  return *this;
}

bool Status::operator==(int code) const { return code_ == code; }

bool Status::operator!=(int code) const { return code_ != code; }

Status::operator int() const { return code_; }

Status::operator bool() const { return code_ == kSuccess; }

int32_t Status::get_err_code() const { return code_; }

const std::string& Status::get_err_msg() const { return msg_; }

void Status::set_err_msg(const std::string& err_msg) { msg_ = err_msg; }

namespace error {

Status Success(const std::string& err_msg) {
  return Status{kSuccess, err_msg};
}

Status FuntionUnImplement(const std::string& err_msg) {
  return Status{kFuntionUnImplement, err_msg};
}

Status PathNotValid(const std::string& err_msg) {
  return Status{kPathNotValid, err_msg};
}

Status ModelParseError(const std::string& err_msg) {
  return Status{kModelParseError, err_msg};
}

Status InternalError(const std::string& err_msg) {
  return Status{kInternalError, err_msg};
}

Status KeyValueHasExist(const std::string& err_msg) {
  return Status{kKeyValueHasExist, err_msg};
}

Status InvalidArgument(const std::string& err_msg) {
  return Status{kInvalidArgument, err_msg};
}

}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.get_err_msg();
  return os;
}

}  // namespace cachehitML

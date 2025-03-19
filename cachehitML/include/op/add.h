#ifndef CACHEHITML_INCLUDE_OP_ADD_H_
#define CACHEHITML_INCLUDE_OP_ADD_H_

#include <base/base.h>
#include <layer.h>

namespace op {
class VecAddLayer : public Layer {
 public:
  explicit VecAddLayer(ML::DeviceType device_type);

  ML::Status check() const override;

  ML::Status forward() override;
};
}  // namespace op
#endif  // CACHEHITML_INCLUDE_OP_ADD_H_
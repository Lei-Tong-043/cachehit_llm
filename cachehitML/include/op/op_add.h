#ifndef CACHEHITML_INCLUDE_OP_ADD_H_
#define CACHEHITML_INCLUDE_OP_ADD_H_

#include "base/base.h"
#include "op_layer.h"

namespace op {
// using ML = cachehitML;
class VecAddLayer : public Layer {
 public:
  explicit VecAddLayer(cachehitML::DeviceType device_type);

  cachehitML::Status check() const override;

  cachehitML::Status forward() override;
};
}  // namespace op
#endif  // CACHEHITML_INCLUDE_OP_ADD_H_
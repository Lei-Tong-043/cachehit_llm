#pragma once

#include <base/base.h>

#include "sampler.h"

namespace sampler {
class ArgmaxSampler : public Sampler {
 public:
  explicit ArgmaxSampler(cachehitML::Device device_type) : Sampler(device_type) {}

  size_t sample(const float* logits, size_t size, void* stream) override;
  
};
}  // namespace sampler
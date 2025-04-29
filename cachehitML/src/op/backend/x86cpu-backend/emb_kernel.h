#pragma once
#include "base/base.h"
#include "op/op_embedding.h"

namespace kernel {
void emb_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t vocab_size, void* stream = nullptr);
}
#pragma once
#include <sentencepiece_processor.h>

#include "layer.h"
#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT)
#include <absl/string/str_join.h>

#endif

namespace op {

class EncodeLayerBase : public Layer {
 public:
  explicit EncodeLayerBase(std::string token_model_path, bool has_bos,
                           bool has_eos)
      : Layer(cachehitML::DeviceType::kDeviceCPU, LayerType::kLayerEncode, "Encode"),
        has_bos_(has_bos),
        has_eos_(has_eos),
        token_model_path_(std::move(token_model_path)) {}

  virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

  virtual std::string decode(int32_t token_id) const = 0;

  virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;

  virtual bool is_sentence_ending(int32_t token_id) const = 0;

  virtual int32_t vocab_size() const = 0;

 protected:
  bool has_bos_ = true;
  bool has_eos_ = false;
  std::string token_model_path_;
};

class SpeEncodeLayer : public EncodeLayerBase {
 public:
  explicit SpeEncodeLayer(std::string token_model_path, bool has_bos,
                          bool has_eos);

  std::vector<int32_t> encode(const std::string& sentence) const override;

  std::vector decode(int32_t token_id) const override;

  std::string decode(const std::vector<int32_t>& token_ids) const override;

  bool is_sentence_ending(int32_t token_id) const override;

  int32_t vocab_size() const override;

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spe_;
};

#if defined (LLAMA3_SUPPORT) || defined (QWEN2_SUPPORT)

/// TODO
/// TODO
/// TODO


#endif


}  // namespace op
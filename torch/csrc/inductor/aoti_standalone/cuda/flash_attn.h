#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/cuda/flash_attn_template.h>

#ifdef __cplusplus
extern "C" {
#endif

using torch::standalone::SlimTensor;

AOTITorchError aoti_torch_cuda__scaled_dot_product_flash_attention(
    AtenTensorHandle query_handle,
    AtenTensorHandle key_handle,
    AtenTensorHandle value_handle,
    double dropout_p,
    int32_t is_causal,
    int32_t return_debug_mask,
    double* scale,
    AtenTensorHandle* ret0,
    AtenTensorHandle* ret1,
    AtenTensorHandle* ret2,
    AtenTensorHandle* ret3,
    int64_t* ret4,
    int64_t* ret5,
    AtenTensorHandle* ret6,
    AtenTensorHandle* ret7,
    AtenTensorHandle* ret8) {
  SlimTensor* query = reinterpret_cast<SlimTensor*>(query_handle);
  SlimTensor* key = reinterpret_cast<SlimTensor*>(key_handle);
  SlimTensor* value = reinterpret_cast<SlimTensor*>(value_handle);

  std::optional<double> scale_opt =
      scale ? std::optional<double>(*scale) : std::nullopt;

  auto
      [attention,
       logsumexp,
       cum_seq_q,
       cum_seq_k,
       max_seqlen_q,
       max_seqlen_k,
       philox_seed,
       philox_offset,
       debug_attn_mask] =
          torch::standalone::_scaled_dot_product_flash_attention_cuda<
              SlimTensor>(
              *query,
              *key,
              *value,
              dropout_p,
              static_cast<bool>(is_causal),
              static_cast<bool>(return_debug_mask),
              scale_opt);

  *ret0 =
      reinterpret_cast<AtenTensorHandle>(new SlimTensor(std::move(attention)));
  *ret1 =
      reinterpret_cast<AtenTensorHandle>(new SlimTensor(std::move(logsumexp)));
  *ret2 =
      reinterpret_cast<AtenTensorHandle>(new SlimTensor(std::move(cum_seq_q)));
  *ret3 =
      reinterpret_cast<AtenTensorHandle>(new SlimTensor(std::move(cum_seq_k)));
  *ret4 = max_seqlen_q;
  *ret5 = max_seqlen_k;
  *ret6 = reinterpret_cast<AtenTensorHandle>(
      new SlimTensor(std::move(philox_seed)));
  *ret7 = reinterpret_cast<AtenTensorHandle>(
      new SlimTensor(std::move(philox_offset)));
  *ret8 = reinterpret_cast<AtenTensorHandle>(
      new SlimTensor(std::move(debug_attn_mask)));

  return AOTI_TORCH_SUCCESS;
}

#ifdef __cplusplus
} // extern "C"
#endif

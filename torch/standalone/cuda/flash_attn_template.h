#pragma once

#include <cmath>
#include <optional>
#include <tuple>

#include <torch/csrc/inductor/aoti_standalone/cuda/flash_attn/flash_api.h>

namespace torch::standalone {

// Forward declaration of _flash_attention_forward
template <typename T>
std::tuple<T, T, T, T, T> _flash_attention_forward(
    const T& query,
    const T& key,
    const T& value,
    const std::optional<T>& cumulative_sequence_length_q,
    const std::optional<T>& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<T>& _seqused_k,
    const std::optional<T>& _alibi_slopes);

template <typename T>
inline float calculate_scale(const T& query, std::optional<double> scale) {
  return scale.has_value()
      ? static_cast<float>(scale.value())
      : static_cast<float>(
            1.0 / std::sqrt(static_cast<double>(query.size(-1))));
}

template <typename T>
std::tuple<T, T, T, T, int64_t, int64_t, T, T, T>
_scaled_dot_product_flash_attention_cuda(
    const T& query,
    const T& key,
    const T& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t max_seqlen_batch_k = key.size(2);
  const int64_t max_seqlen_batch_v = value.size(2);

  TORCH_CHECK(
      max_seqlen_batch_k == max_seqlen_batch_v,
      "Key and Value must have the same sequence length");

  // Query -> Query(Batch x Q_seq_len  x Num_heads x Dim_per_head)
  // Key   -> Key  (Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value -> Value(Batch x KV_seq_len x Num_heads x Dim_per_head)
  T q_t = query.transpose(1, 2);
  T k_t = key.transpose(1, 2);
  T v_t = value.transpose(1, 2);

  auto [output, logsumexp, philox_seed, philox_offset, debug_attn_mask] =
      _flash_attention_forward<T>(
          q_t,
          k_t,
          v_t,
          std::optional<T>{},
          std::optional<T>{},
          max_seqlen_batch_q,
          max_seqlen_batch_k,
          dropout_p,
          is_causal,
          return_debug_mask,
          scale,
          std::nullopt,
          std::nullopt,
          std::optional<T>{},
          std::optional<T>{});

  T attention = output.transpose(1, 2);

  auto cum_seq_q = create_empty_tensor({}, {}, query.dtype(), query.device());
  auto cum_seq_k = create_empty_tensor({}, {}, key.dtype(), key.device());

  return std::make_tuple(
      std::move(attention),
      std::move(logsumexp),
      std::move(cum_seq_q),
      std::move(cum_seq_k),
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      std::move(philox_seed),
      std::move(philox_offset),
      std::move(debug_attn_mask));
}

template <typename T>
std::tuple<T, T, T, T, T> _flash_attention_forward(
    const T& query,
    const T& key,
    const T& value,
    const std::optional<T>& cumulative_sequence_length_q,
    const std::optional<T>& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<T>& _seqused_k,
    const std::optional<T>& _alibi_slopes) {
  const auto softmax_scale = calculate_scale<T>(query, scale);

  std::optional<T> out = std::nullopt;

  std::optional<T> seqused_k = _seqused_k;
  std::optional<T> block_table =
      std::nullopt; // we are not using the block table yet
  std::optional<T> alibi_slopes = _alibi_slopes;
  const float softcap = 0.0;

  const int64_t non_null_window_left =
      window_size_left.has_value() ? *window_size_left : -1;
  const int64_t non_null_window_right =
      window_size_right.has_value() ? *window_size_right : -1;

  TORCH_CHECK(
      cumulative_sequence_length_q.has_value() ==
          cumulative_sequence_length_k.has_value(),
      "cumulative_sequence_length_q and cumulative_sequence_length_k must be both set or both not set");

  T output = create_empty_tensor({}, {}, query.dtype(), query.device());
  T q_padded = create_empty_tensor({}, {}, query.dtype(), query.device());
  T k_padded = create_empty_tensor({}, {}, key.dtype(), key.device());
  T v_padded = create_empty_tensor({}, {}, value.dtype(), value.device());
  T logsumexp = create_empty_tensor({}, {}, query.dtype(), query.device());
  T output_shape = create_empty_tensor({}, {}, query.dtype(), query.device());
  T philox_seed = create_empty_tensor({}, {}, query.dtype(), query.device());
  T philox_offset = create_empty_tensor({}, {}, query.dtype(), query.device());
  T debug_attn_mask =
      create_empty_tensor({}, {}, query.dtype(), query.device());

  if (cumulative_sequence_length_q.has_value()) {
    std::tie(
        output,
        q_padded,
        k_padded,
        v_padded,
        logsumexp,
        philox_seed,
        philox_offset,
        debug_attn_mask) =
        mha_varlen_fwd(
            query,
            key,
            value,
            out,
            cumulative_sequence_length_q.value(),
            cumulative_sequence_length_k.value(),
            seqused_k, /*seqused_k*/
            block_table, /*block_table*/
            alibi_slopes, /*alibi_slopes*/
            max_seqlen_batch_q,
            max_seqlen_batch_k,
            dropout_p,
            softmax_scale,
            false /*zero_tensors*/,
            is_causal,
            non_null_window_left,
            non_null_window_right,
            softcap,
            return_debug_mask);
  } else {
    std::tie(
        output,
        q_padded,
        k_padded,
        v_padded,
        logsumexp,
        philox_seed,
        philox_offset,
        debug_attn_mask) =
        mha_fwd(
            query,
            key,
            value,
            out,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            is_causal,
            non_null_window_left,
            non_null_window_right,
            softcap,
            return_debug_mask /*return_softmax (this is used for testing)*/);
  }
  debug_attn_mask = return_debug_mask
      ? debug_attn_mask
      : create_empty_tensor({}, {}, query.dtype(), query.device());
  return std::make_tuple(
      std::move(output),
      std::move(logsumexp),
      std::move(philox_seed),
      std::move(philox_offset),
      std::move(debug_attn_mask));
}

} // namespace torch::standalone

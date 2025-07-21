#pragma once
#include <cstddef>
#include <optional>
#include <tuple>

#include <c10/util/Exception.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

using torch::standalone::SlimTensor;

std::tuple<
    SlimTensor,
    SlimTensor,
    SlimTensor,
    SlimTensor,
    SlimTensor,
    SlimTensor,
    SlimTensor,
    SlimTensor>
mha_fwd(
    const SlimTensor& q, // batch_size x seqlen_q x num_heads x head_size
    const SlimTensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const SlimTensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<SlimTensor>&
        out_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<SlimTensor>&
        alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax);

std::tuple<
    SlimTensor,
    SlimTensor,
    SlimTensor,
    SlimTensor,
    SlimTensor,
    SlimTensor,
    SlimTensor,
    SlimTensor>
mha_varlen_fwd(
    const SlimTensor&
        q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const SlimTensor&
        k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const SlimTensor&
        v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<SlimTensor>&
        out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const SlimTensor& cu_seqlens_q, // b+1
    const SlimTensor& cu_seqlens_k, // b+1
    std::optional<SlimTensor>&
        seqused_k, // b. If given, only this many elements of each batch
                   // element's keys are used.
    std::optional<SlimTensor>&
        block_table_, // batch_size x max_num_blocks_per_seq
    std::optional<SlimTensor>& alibi_slopes_, // num_heads or b x num_heads
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax);

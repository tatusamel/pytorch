#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

TEST(FlashAttentionTest, BasicATenOperator) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  // Create test tensors on CUDA
  at::Device cuda_device(at::kCUDA);
  auto options = at::TensorOptions().dtype(at::kHalf).device(cuda_device);

  int64_t batch_size = 2;
  int64_t seq_len = 128;
  int64_t head_dim = 64;
  int64_t num_heads = 8;

  // Create query, key, value tensors
  at::Tensor query = at::randn({batch_size, num_heads, seq_len, head_dim}, options);
  at::Tensor key = at::randn({batch_size, num_heads, seq_len, head_dim}, options);
  at::Tensor value = at::randn({batch_size, num_heads, seq_len, head_dim}, options);

  // Test basic call to _scaled_dot_product_flash_attention
  try {
    auto [result, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask] = at::_scaled_dot_product_flash_attention(
        query, key, value,
        /*dropout_p=*/0.0,
        /*is_causal=*/false,
        /*return_debug_mask=*/false,
        /*scale=*/std::nullopt);

    // Basic verification - check that result has expected shape
    EXPECT_EQ(result.dim(), 4);
    EXPECT_EQ(result.size(0), batch_size);
    EXPECT_EQ(result.size(1), num_heads);
    EXPECT_EQ(result.size(2), seq_len);
    EXPECT_EQ(result.size(3), head_dim);
    EXPECT_TRUE(result.device().is_cuda());
    EXPECT_EQ(result.dtype(), at::kHalf);

    // Check that result is not all zeros or NaN
    EXPECT_FALSE(at::all(result == 0).item<bool>());
    EXPECT_FALSE(at::any(at::isnan(result)).item<bool>());

    std::cout << "Flash attention test passed - result shape: "
              << result.sizes() << std::endl;

  } catch (const std::exception& e) {
    FAIL() << "Flash attention operator failed with error: " << e.what();
  }
}

TEST(FlashAttentionTest, CausalMask) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  at::Device cuda_device(at::kCUDA);
  auto options = at::TensorOptions().dtype(at::kHalf).device(cuda_device);

  int64_t batch_size = 1;
  int64_t seq_len = 64;
  int64_t head_dim = 32;
  int64_t num_heads = 4;

  at::Tensor query = at::randn({batch_size, num_heads, seq_len, head_dim}, options);
  at::Tensor key = at::randn({batch_size, num_heads, seq_len, head_dim}, options);
  at::Tensor value = at::randn({batch_size, num_heads, seq_len, head_dim}, options);

  try {
    // Test with causal mask
    auto [result_causal, logsumexp_causal, cum_seq_q_causal, cum_seq_k_causal, max_q_causal, max_k_causal, philox_seed_causal, philox_offset_causal, debug_attn_mask_causal] = at::_scaled_dot_product_flash_attention(
        query, key, value,
        /*dropout_p=*/0.0,
        /*is_causal=*/true,
        /*return_debug_mask=*/false,
        /*scale=*/std::nullopt);

    // Test without causal mask
    auto [result_no_causal, logsumexp_no_causal, cum_seq_q_no_causal, cum_seq_k_no_causal, max_q_no_causal, max_k_no_causal, philox_seed_no_causal, philox_offset_no_causal, debug_attn_mask_no_causal] = at::_scaled_dot_product_flash_attention(
        query, key, value,
        /*dropout_p=*/0.0,
        /*is_causal=*/false,
        /*return_debug_mask=*/false,
        /*scale=*/std::nullopt);

    // Both should have same shape
    EXPECT_EQ(result_causal.sizes(), result_no_causal.sizes());

    // Results should be different (causal vs non-causal)
    EXPECT_FALSE(at::allclose(result_causal, result_no_causal));

    std::cout << "Causal mask test passed" << std::endl;

  } catch (const std::exception& e) {
    FAIL() << "Causal mask test failed with error: " << e.what();
  }
}

TEST(FlashAttentionTest, CustomScale) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  at::Device cuda_device(at::kCUDA);
  auto options = at::TensorOptions().dtype(at::kHalf).device(cuda_device);

  int64_t batch_size = 1;
  int64_t seq_len = 32;
  int64_t head_dim = 16;
  int64_t num_heads = 2;

  at::Tensor query = at::randn({batch_size, num_heads, seq_len, head_dim}, options);
  at::Tensor key = at::randn({batch_size, num_heads, seq_len, head_dim}, options);
  at::Tensor value = at::randn({batch_size, num_heads, seq_len, head_dim}, options);

  try {
    // Test with custom scale
    double custom_scale = 0.5;
    auto [result_scaled, logsumexp_scaled, cum_seq_q_scaled, cum_seq_k_scaled, max_q_scaled, max_k_scaled, philox_seed_scaled, philox_offset_scaled, debug_attn_mask_scaled] = at::_scaled_dot_product_flash_attention(
        query, key, value,
        /*dropout_p=*/0.0,
        /*is_causal=*/false,
        /*return_debug_mask=*/false,
        /*scale=*/custom_scale);

    // Test with default scale (should be 1/sqrt(head_dim))
    auto [result_default, logsumexp_default, cum_seq_q_default, cum_seq_k_default, max_q_default, max_k_default, philox_seed_default, philox_offset_default, debug_attn_mask_default] = at::_scaled_dot_product_flash_attention(
        query, key, value,
        /*dropout_p=*/0.0,
        /*is_causal=*/false,
        /*return_debug_mask=*/false,
        /*scale=*/std::nullopt);

    // Both should have same shape
    EXPECT_EQ(result_scaled.sizes(), result_default.sizes());

    // Results should be different due to different scaling
    EXPECT_FALSE(at::allclose(result_scaled, result_default));

    std::cout << "Custom scale test passed" << std::endl;

  } catch (const std::exception& e) {
    FAIL() << "Custom scale test failed with error: " << e.what();
  }
}

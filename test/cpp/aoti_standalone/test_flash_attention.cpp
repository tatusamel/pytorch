#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/torch.h>
#include <optional>

// Guard for CUDA specific code and Flash Attention availability
#if defined(USE_CUDA)
#include <torch/csrc/inductor/aoti_standalone/cuda/flash_attn.h>
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

using ::testing::ElementsAreArray;
using torch::standalone::SlimTensor;

// Helper to convert a SlimTensor (potentially on CUDA) to a CPU at::Tensor for
// easy comparison. It clones the data to ensure the at::Tensor owns its memory.
at::Tensor slim_to_cpu_aten(SlimTensor* slim_tensor) {
  if (!slim_tensor || slim_tensor->numel() == 0) {
    // Return an empty tensor if the slim tensor is null or has no elements.
    return at::empty(
        {0},
        at::TensorOptions().dtype(
            slim_tensor ? slim_tensor->dtype() : at::kFloat));
  }
  SlimTensor slim_cpu = slim_tensor->to(at::kCPU);
  return at::from_blob(
             slim_cpu.data_ptr(),
             slim_cpu.sizes(),
             slim_cpu.strides(),
             slim_cpu.dtype())
      .clone(); // Crucial: clone to own the data
}

// Define a test fixture for convenience.
// This sets up the CUDA device and tensor options for all tests in this suite.
class ScaledDotProductFlashAttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test suite.";
    }
    cuda_device_ = at::Device(at::kCUDA);
    // FlashAttention is typically optimized for half precision.
    options_ = at::TensorOptions().dtype(at::kHalf).device(cuda_device_);
  }

  // Helper to create both at::Tensor and its corresponding SlimTensor view.
  void create_tensors(
      const std::vector<int64_t>& q_shape,
      const std::vector<int64_t>& kv_shape,
      at::Tensor& at_q,
      std::optional<SlimTensor>& slim_q,
      at::Tensor& at_k,
      std::optional<SlimTensor>& slim_k,
      at::Tensor& at_v,
      std::optional<SlimTensor>& slim_v) {
    at_q = at::randn(q_shape, options_);
    at_k = at::randn(kv_shape, options_);
    at_v = at::randn(kv_shape, options_);

    slim_q = torch::standalone::create_tensor_from_blob(
        at_q.data_ptr(),
        at_q.sizes(),
        at_q.strides(),
        options_.dtype().toScalarType(),
        cuda_device_);
    slim_k = torch::standalone::create_tensor_from_blob(
        at_k.data_ptr(),
        at_k.sizes(),
        at_k.strides(),
        options_.dtype().toScalarType(),
        cuda_device_);
    slim_v = torch::standalone::create_tensor_from_blob(
        at_v.data_ptr(),
        at_v.sizes(),
        at_v.strides(),
        options_.dtype().toScalarType(),
        cuda_device_);
  }

  // The main verification function that runs both implementations and compares
  // results.
  void run_and_verify(
      SlimTensor& slim_query,
      SlimTensor& slim_key,
      SlimTensor& slim_value,
      const at::Tensor& at_query,
      const at::Tensor& at_key,
      const at::Tensor& at_value,
      double dropout_p,
      bool is_causal,
      bool return_debug_mask,
      std::optional<double> scale) {
    // AOTI shim call
    AtenTensorHandle ret0 = nullptr, ret1 = nullptr, ret2 = nullptr,
                     ret3 = nullptr;
    AtenTensorHandle ret6 = nullptr, ret7 = nullptr, ret8 = nullptr;
    int64_t ret4 = 0, ret5 = 0;

    AOTITorchError err = aoti_torch_cuda__scaled_dot_product_flash_attention(
        reinterpret_cast<AtenTensorHandle>(&slim_query),
        reinterpret_cast<AtenTensorHandle>(&slim_key),
        reinterpret_cast<AtenTensorHandle>(&slim_value),
        dropout_p,
        is_causal,
        return_debug_mask,
        scale.has_value() ? &scale.value() : nullptr,
        &ret0,
        &ret1,
        &ret2,
        &ret3,
        &ret4,
        &ret5,
        &ret6,
        &ret7,
        &ret8);

    ASSERT_EQ(err, AOTI_TORCH_SUCCESS);

    // PyTorch reference call
    auto at_result_tuple = at::_scaled_dot_product_flash_attention(
        at_query,
        at_key,
        at_value,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale);

    // Unpack results for comparison
    auto at_attention = std::get<0>(at_result_tuple);
    auto at_logsumexp = std::get<1>(at_result_tuple);

    auto* slim_attention = reinterpret_cast<SlimTensor*>(ret0);
    auto* slim_logsumexp = reinterpret_cast<SlimTensor*>(ret1);
    ASSERT_NE(slim_attention, nullptr);
    ASSERT_NE(slim_logsumexp, nullptr);

    // 1. Compare metadata (sequence lengths)

    ASSERT_EQ(ret4, std::get<4>(at_result_tuple));
    ASSERT_EQ(ret5, std::get<5>(at_result_tuple));

    // 2. Compare main output tensors (attention and logsumexp)
    at::Tensor slim_attention_aten = slim_to_cpu_aten(slim_attention);
    at::Tensor slim_logsumexp_aten = slim_to_cpu_aten(slim_logsumexp);
    at::Tensor at_attention_cpu = at_attention.to(at::kCPU);
    at::Tensor at_logsumexp_cpu = at_logsumexp.to(at::kCPU);

    EXPECT_THAT(
        slim_attention->sizes(), ElementsAreArray(at_attention.sizes()));
    EXPECT_THAT(
        slim_logsumexp->sizes(), ElementsAreArray(at_logsumexp.sizes()));

    // Use allclose for float comparisons with a reasonable tolerance for flash
    // attention.
    ASSERT_TRUE(
        at::allclose(slim_attention_aten, at_attention_cpu, 1e-3, 1e-3));
    ASSERT_TRUE(
        at::allclose(slim_logsumexp_aten, at_logsumexp_cpu, 1e-3, 1e-3));

    // 3. Compare debug mask if requested
    if (return_debug_mask) {
      auto* slim_debug_mask = reinterpret_cast<SlimTensor*>(ret8);
      auto at_debug_mask = std::get<8>(at_result_tuple);
      ASSERT_NE(slim_debug_mask, nullptr);

      EXPECT_THAT(
          slim_debug_mask->sizes(), ElementsAreArray(at_debug_mask.sizes()));
    }

    // 4. Cleanup all returned tensor handles
    delete reinterpret_cast<SlimTensor*>(ret0);
    delete reinterpret_cast<SlimTensor*>(ret1);
    delete reinterpret_cast<SlimTensor*>(ret2);
    delete reinterpret_cast<SlimTensor*>(ret3);
    delete reinterpret_cast<SlimTensor*>(ret6);
    delete reinterpret_cast<SlimTensor*>(ret7);
    delete reinterpret_cast<SlimTensor*>(ret8);
  }

  at::Device cuda_device_ = at::Device(at::kCPU);
  at::TensorOptions options_;
};

// Test Case 1: Standard causal attention for language modeling.
TEST_F(ScaledDotProductFlashAttentionTest, CausalMaskNoDropout) {
  const int64_t batch_size = 4;
  const int64_t num_heads = 8;
  const int64_t seq_len = 512;
  const int64_t head_dim = 64;

  at::Tensor at_q, at_k, at_v;
  std::optional<SlimTensor> slim_q, slim_k, slim_v;
  create_tensors(
      {batch_size, num_heads, seq_len, head_dim},
      {batch_size, num_heads, seq_len, head_dim},
      at_q,
      slim_q,
      at_k,
      slim_k,
      at_v,
      slim_v);

  run_and_verify(
      *slim_q,
      *slim_k,
      *slim_v,
      at_q,
      at_k,
      at_v,
      0.0, // dropout_p (must be 0 for deterministic comparison)
      true, // is_causal
      false, // return_debug_mask
      std::nullopt // scale
  );
}

// Test Case 2: Non-causal attention (e.g., for an encoder) with a custom scale
// factor.
TEST_F(ScaledDotProductFlashAttentionTest, NonCausalWithScale) {
  const int64_t batch_size = 2;
  const int64_t num_heads = 4;
  const int64_t q_seq_len = 256;
  const int64_t kv_seq_len = 128;
  const int64_t head_dim = 128;

  at::Tensor at_q, at_k, at_v;
  std::optional<SlimTensor> slim_q, slim_k, slim_v;
  create_tensors(
      {batch_size, num_heads, q_seq_len, head_dim},
      {batch_size, num_heads, kv_seq_len, head_dim},
      at_q,
      slim_q,
      at_k,
      slim_k,
      at_v,
      slim_v);

  const double custom_scale = 0.125;
  run_and_verify(
      *slim_q,
      *slim_k,
      *slim_v,
      at_q,
      at_k,
      at_v,
      0.0, // dropout_p
      false, // is_causal
      false, // return_debug_mask
      custom_scale);
}

// Test Case 3: Causal attention with non-square sequence lengths, common in
// text generation.
TEST_F(ScaledDotProductFlashAttentionTest, CausalNonSquare) {
  const int64_t batch_size = 2;
  const int64_t num_heads = 8;
  const int64_t q_seq_len = 1; // A single new token
  const int64_t kv_seq_len = 1024; // Against a long key/value cache
  const int64_t head_dim = 64;

  at::Tensor at_q, at_k, at_v;
  std::optional<SlimTensor> slim_q, slim_k, slim_v;
  create_tensors(
      {batch_size, num_heads, q_seq_len, head_dim},
      {batch_size, num_heads, kv_seq_len, head_dim},
      at_q,
      slim_q,
      at_k,
      slim_k,
      at_v,
      slim_v);

  run_and_verify(
      *slim_q,
      *slim_k,
      *slim_v,
      at_q,
      at_k,
      at_v,
      0.0, // dropout_p
      true, // is_causal
      false, // return_debug_mask
      std::nullopt);
}

// Test Case 4: Non-causal attention with non-square sequence lengths, common in
// encoder-decoder architectures.
TEST_F(ScaledDotProductFlashAttentionTest, NonCausalNonSquare) {
  const int64_t batch_size = 3;
  const int64_t num_heads = 6;
  const int64_t q_seq_len = 512; // Query sequence length
  const int64_t kv_seq_len = 256; // Key/Value sequence length
  const int64_t head_dim = 64;

  at::Tensor at_q, at_k, at_v;
  std::optional<SlimTensor> slim_q, slim_k, slim_v;
  create_tensors(
      {batch_size, num_heads, q_seq_len, head_dim},
      {batch_size, num_heads, kv_seq_len, head_dim},
      at_q,
      slim_q,
      at_k,
      slim_k,
      at_v,
      slim_v);

  run_and_verify(
      *slim_q,
      *slim_k,
      *slim_v,
      at_q,
      at_k,
      at_v,
      0.0, // dropout_p
      false, // is_causal
      false, // return_debug_mask
      std::nullopt);
}

#endif // USE_CUDA

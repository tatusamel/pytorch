#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/torch.h>

#include <torch/csrc/inductor/aoti_standalone/cpu/transpose_int.h>
#include <torch/csrc/inductor/aoti_standalone/cuda/transpose_int.h>

using ::testing::ElementsAreArray;

TEST(TransposeTest, TransposeOp) {
  // crate ground-truth aten tensor
  at::Tensor at_tensor = at::arange(6, at::kFloat).reshape({2, 3});
  ASSERT_TRUE(at_tensor.is_contiguous());

  // create a non-owning SlimTensor view of the ATen tensor
  torch::standalone::SlimTensor slim_tensor_self =
      torch::standalone::create_tensor_from_blob(
          at_tensor.data_ptr(),
          c10::IntArrayRef(at_tensor.sizes().data(), at_tensor.dim()),
          c10::IntArrayRef(at_tensor.strides().data(), at_tensor.dim()),
          at::kFloat);

  AtenTensorHandle result_handle = nullptr;
  AOTITorchError err = aoti_torch_cpu_transpose_int(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor_self),
      0,
      1,
      &result_handle);
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);

  // perform the operation on the ATen tensor for ground truth
  at::Tensor at_result = at::transpose(at_tensor, 0, 1);

  // convert our result handle back to a SlimTensor for inspection
  torch::standalone::SlimTensor* slim_result =
      reinterpret_cast<torch::standalone::SlimTensor*>(result_handle);

  ASSERT_NE(slim_result, nullptr);

  EXPECT_EQ(slim_result->dim(), at_result.dim());
  EXPECT_THAT(slim_result->sizes(), ElementsAreArray(at_result.sizes()));
  EXPECT_THAT(slim_result->strides(), ElementsAreArray(at_result.strides()));
  EXPECT_EQ(slim_result->numel(), at_result.numel());
  // A transposed tensor is not contiguous
  EXPECT_FALSE(slim_result->is_contiguous());

  // verify it's a view by cheking that the data pointer is unchanged
  EXPECT_EQ(slim_result->data_ptr(), at_tensor.data_ptr());

  delete slim_result;
}

// Test case for a no-op transpose where dim0 == dim1
TEST(TransposeTest, TransposeNoOp) {
  at::Tensor at_tensor = at::arange(6, at::kFloat).reshape({2, 3});
  torch::standalone::SlimTensor slim_tensor_self =
      torch::standalone::create_tensor_from_blob(
          at_tensor.data_ptr(),
          c10::IntArrayRef(at_tensor.sizes().data(), at_tensor.dim()),
          c10::IntArrayRef(at_tensor.strides().data(), at_tensor.dim()),
          at::kFloat);

  AtenTensorHandle result_handle = nullptr;
  AOTITorchError err = aoti_torch_cpu_transpose_int(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor_self),
      1,
      1, // No-op transpose
      &result_handle);
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);
  torch::standalone::SlimTensor* slim_result =
      reinterpret_cast<torch::standalone::SlimTensor*>(result_handle);

  // A no-op transpose should return a view with identical metadata
  ASSERT_NE(slim_result, nullptr);
  EXPECT_THAT(slim_result->sizes(), ElementsAreArray(at_tensor.sizes()));
  EXPECT_THAT(slim_result->strides(), ElementsAreArray(at_tensor.strides()));
  EXPECT_TRUE(slim_result->is_contiguous());
  EXPECT_EQ(slim_result->data_ptr(), at_tensor.data_ptr());

  delete slim_result;
}

// Test case that transposing twice returns the original tensor view
TEST(TransposeTest, TransposeOfTranspose) {
  at::Tensor at_tensor = at::arange(6, at::kFloat).reshape({2, 3});
  torch::standalone::SlimTensor slim_tensor_self =
      torch::standalone::create_tensor_from_blob(
          at_tensor.data_ptr(),
          c10::IntArrayRef(at_tensor.sizes().data(), at_tensor.dim()),
          c10::IntArrayRef(at_tensor.strides().data(), at_tensor.dim()),
          at::kFloat);

  // First transpose
  AtenTensorHandle t1_handle = nullptr;

  AOTITorchError err = aoti_torch_cpu_transpose_int(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor_self), 0, 1, &t1_handle);
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);

  torch::standalone::SlimTensor* t1_result =
      reinterpret_cast<torch::standalone::SlimTensor*>(t1_handle);
  EXPECT_FALSE(t1_result->is_contiguous());

  // Second transpose (should return to original)
  AtenTensorHandle t2_handle = nullptr;

  err = aoti_torch_cpu_transpose_int(t1_handle, 0, 1, &t2_handle);
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);
  torch::standalone::SlimTensor* t2_result =
      reinterpret_cast<torch::standalone::SlimTensor*>(t2_handle);

  // Verify it matches the original tensor
  ASSERT_NE(t2_result, nullptr);
  EXPECT_THAT(t2_result->sizes(), ElementsAreArray(at_tensor.sizes()));
  EXPECT_THAT(t2_result->strides(), ElementsAreArray(at_tensor.strides()));
  EXPECT_TRUE(t2_result->is_contiguous());

  delete t1_result;
  delete t2_result;
}

#if defined(USE_CUDA)
TEST(TransposeTest, TransposeOpCuda) {
  // 1. Setup: Check if a CUDA device is actually available at runtime
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  at::Device cuda_device(at::kCUDA);

  // Create the ground-truth tensor on the CUDA device
  at::Tensor at_tensor =
      at::arange(6, at::TensorOptions().dtype(at::kFloat).device(cuda_device))
          .reshape({2, 3});

  // Create a non-owning SlimTensor view of the ATen tensor on CUDA
  torch::standalone::SlimTensor slim_tensor_self =
      torch::standalone::create_tensor_from_blob(
          at_tensor.data_ptr(),
          c10::IntArrayRef(at_tensor.sizes().data(), at_tensor.dim()),
          c10::IntArrayRef(at_tensor.strides().data(), at_tensor.dim()),
          at::kFloat,
          cuda_device);

  // 2. Action
  AtenTensorHandle result_handle = nullptr;

  AOTITorchError err = aoti_torch_cuda_transpose_int(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor_self),
      0,
      1,
      &result_handle);

  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);

  at::Tensor at_result = at::transpose(at_tensor, 0, 1);
  torch::standalone::SlimTensor* slim_result =
      reinterpret_cast<torch::standalone::SlimTensor*>(result_handle);

  // 3. Verify Results
  ASSERT_NE(slim_result, nullptr);
  EXPECT_TRUE(slim_result->device().is_cuda());
  EXPECT_THAT(slim_result->sizes(), ElementsAreArray(at_result.sizes()));
  EXPECT_THAT(slim_result->strides(), ElementsAreArray(at_result.strides()));
  EXPECT_EQ(slim_result->data_ptr(), at_tensor.data_ptr());

  delete slim_result;
}
#endif // USE_CUDA

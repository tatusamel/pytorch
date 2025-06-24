#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/torch.h>

#include <torch/csrc/inductor/aoti_standalone/cpu/transpose_int.h>
#include <torch/csrc/inductor/aoti_standalone/cuda/transpose_int.h>

using ::testing::ElementsAreArray;

TEST(Transpose, TransposeBasic) {
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

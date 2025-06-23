#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/csrc/inductor/aoti_standalone/cpu/scalar_tensor.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/torch.h>
#include <cstdint>

namespace torch::standalone {

// Test case for a float scalar tensor
TEST(ScalarTensorTest, ScalarTensorOp) {
  // Define the scalar and options
  const double scalar_value = 3.14;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);

  // Create the ground-truth tensor using the ATen implementation
  at::Tensor at_tensor = at::scalar_tensor(scalar_value, options);

  // Call our shim implementation
  AtenTensorHandle slim_tensor_handle = nullptr;
  auto dtype = static_cast<int32_t>(options.dtype().toScalarType());
  auto layout = static_cast<int32_t>(options.layout());
  auto device = static_cast<int32_t>(options.device().type());
  auto device_index = static_cast<int32_t>(options.device().index());
  auto pin_memory = static_cast<int32_t>(options.pinned_memory());

  AOTITorchError err = aoti_torch_cpu_scalar_tensor(
      scalar_value,
      &dtype,
      &layout,
      &device,
      device_index,
      &pin_memory,
      &slim_tensor_handle);

  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);

  SlimTensor* slim_tensor = reinterpret_cast<SlimTensor*>(slim_tensor_handle);

  // Check metadata
  ASSERT_NE(slim_tensor, nullptr);
  EXPECT_EQ(slim_tensor->dim(), 0);
  EXPECT_EQ(slim_tensor->numel(), 1);
  EXPECT_EQ(slim_tensor->dtype(), at_tensor.dtype());

  // Check the actual value
  float slim_data_value = *static_cast<float*>(slim_tensor->data_ptr());
  float at_data_value = *static_cast<float*>(at_tensor.data_ptr());
  EXPECT_FLOAT_EQ(slim_data_value, at_data_value);
  EXPECT_FLOAT_EQ(slim_data_value, static_cast<float>(scalar_value));

  delete slim_tensor;
}

// Test case for a long (int64_t) scalar, verifying truncation.
TEST(ScalarTensorTest, ScalarTensorOpLong) {
  const double scalar_value = -55.9;
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  at::Tensor at_tensor = at::scalar_tensor(scalar_value, options);

  AtenTensorHandle slim_tensor_handle = nullptr;
  auto dtype = static_cast<int32_t>(options.dtype().toScalarType());
  auto device = static_cast<int32_t>(options.device().type());

  AOTITorchError err = aoti_torch_cpu_scalar_tensor(
      scalar_value, &dtype, nullptr, &device, 0, nullptr, &slim_tensor_handle);

  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);

  SlimTensor* slim_tensor = reinterpret_cast<SlimTensor*>(slim_tensor_handle);

  ASSERT_NE(slim_tensor, nullptr);
  EXPECT_EQ(slim_tensor->numel(), 1);
  EXPECT_EQ(slim_tensor->dtype(), at::kLong);

  int64_t slim_data_value = *static_cast<int64_t*>(slim_tensor->data_ptr());
  int64_t at_data_value = *static_cast<int64_t*>(at_tensor.data_ptr());
  EXPECT_EQ(slim_data_value, at_data_value);
  EXPECT_EQ(slim_data_value, -55L); // Verify truncation behavior

  delete slim_tensor;
}

// Test case for a bool scalar from a non-zero value.
TEST(ScalarTensorTest, ScalarTensorOpBoolTrue) {
  const double scalar_value = -100.0; // Non-zero
  auto options = at::TensorOptions().dtype(at::kBool).device(at::kCPU);
  at::Tensor at_tensor = at::scalar_tensor(scalar_value, options);

  AtenTensorHandle slim_tensor_handle = nullptr;
  auto dtype = static_cast<int32_t>(options.dtype().toScalarType());
  auto device = static_cast<int32_t>(options.device().type());

  AOTITorchError err = aoti_torch_cpu_scalar_tensor(
      scalar_value, &dtype, nullptr, &device, 0, nullptr, &slim_tensor_handle);

  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);
  SlimTensor* slim_tensor = reinterpret_cast<SlimTensor*>(slim_tensor_handle);

  ASSERT_NE(slim_tensor, nullptr);
  EXPECT_EQ(slim_tensor->dtype(), at::kBool);

  bool slim_data_value = *static_cast<bool*>(slim_tensor->data_ptr());
  bool at_data_value = *static_cast<bool*>(at_tensor.data_ptr());
  EXPECT_EQ(slim_data_value, at_data_value);
  EXPECT_EQ(slim_data_value, true);

  delete slim_tensor;
}

// Test case for a bool scalar from a zero value.
TEST(ScalarTensorTest, ScalarTensorOpBoolFalse) {
  const double scalar_value = 0.0;
  auto options = at::TensorOptions().dtype(at::kBool).device(at::kCPU);
  at::Tensor at_tensor = at::scalar_tensor(scalar_value, options);

  AtenTensorHandle slim_tensor_handle = nullptr;
  auto dtype = static_cast<int32_t>(options.dtype().toScalarType());
  auto device = static_cast<int32_t>(options.device().type());

  AOTITorchError err = aoti_torch_cpu_scalar_tensor(
      scalar_value, &dtype, nullptr, &device, 0, nullptr, &slim_tensor_handle);

  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);
  SlimTensor* slim_tensor = reinterpret_cast<SlimTensor*>(slim_tensor_handle);

  ASSERT_NE(slim_tensor, nullptr);
  EXPECT_EQ(slim_tensor->dtype(), at::kBool);

  bool slim_data_value = *static_cast<bool*>(slim_tensor->data_ptr());
  bool at_data_value = *static_cast<bool*>(at_tensor.data_ptr());
  EXPECT_EQ(slim_data_value, at_data_value);
  EXPECT_EQ(slim_data_value, false);

  delete slim_tensor;
}

} // namespace torch::standalone

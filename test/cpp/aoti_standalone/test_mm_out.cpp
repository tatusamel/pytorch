#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

// Guard for CUDA specific code
#if defined(USE_CUDA)
#include <torch/csrc/inductor/aoti_standalone/cuda/mm_out.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include "test_utils.h"

using ::testing::ElementsAreArray;
using torch::standalone::SlimTensor;

// Define a test fixture for convenience.
class MmOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test suite.";
    }
    cuda_device_ = at::Device(at::kCUDA);
  }

  // The main verification function that runs both implementations and compares
  // results.
  void run_and_verify(const at::Tensor& at_mat1, const at::Tensor& at_mat2) {
    // Create SlimTensor views of the at::Tensors
    SlimTensor slim_mat1 = torch::standalone::create_tensor_from_blob(
        at_mat1.data_ptr(),
        at_mat1.sizes(),
        at_mat1.strides(),
        at_mat1.scalar_type(),
        cuda_device_);
    SlimTensor slim_mat2 = torch::standalone::create_tensor_from_blob(
        at_mat2.data_ptr(),
        at_mat2.sizes(),
        at_mat2.strides(),
        at_mat2.scalar_type(),
        cuda_device_);

    // Create output tensors for both implementations.
    at::Tensor at_out =
        at::empty({at_mat1.size(0), at_mat2.size(1)}, at_mat1.options());
    SlimTensor slim_out = torch::standalone::create_empty_tensor(
        at_out.sizes(), at_out.strides(), at_out.scalar_type(), cuda_device_);

    // --- AOTI shim call ---
    AOTITorchError err = aoti_torch_cuda_mm_out(
        reinterpret_cast<AtenTensorHandle>(&slim_out),
        reinterpret_cast<AtenTensorHandle>(&slim_mat1),
        reinterpret_cast<AtenTensorHandle>(&slim_mat2));
    ASSERT_EQ(err, AOTI_TORCH_SUCCESS);

    // --- PyTorch reference call ---
    at::mm_out(at_out, at_mat1, at_mat2);

    // --- Verification ---
    at::Tensor slim_out_aten_cpu = slim_to_cpu_aten(slim_out);
    at::Tensor at_out_cpu = at_out.to(at::kCPU);

    // Check shapes
    EXPECT_THAT(slim_out.sizes(), ElementsAreArray(at_out.sizes()));

    // Use allclose for float comparisons with a reasonable tolerance.
    ASSERT_TRUE(at::allclose(slim_out_aten_cpu, at_out_cpu, 1e-3, 1e-3));
  }

  at::Device cuda_device_ = at::Device(at::kCPU);
};

// Test Case 1: Standard square matrix multiplication with floats
TEST_F(MmOutTest, StandardFloat) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device_);
  at::Tensor at_mat1 = at::randn({50, 100}, options);
  at::Tensor at_mat2 = at::randn({100, 75}, options);
  run_and_verify(at_mat1, at_mat2);
}

// Test Case 2: Test with half-precision tensors
TEST_F(MmOutTest, StandardHalf) {
  auto options = at::TensorOptions().dtype(at::kHalf).device(cuda_device_);
  at::Tensor at_mat1 = at::randn({32, 64}, options);
  at::Tensor at_mat2 = at::randn({64, 32}, options);
  run_and_verify(at_mat1, at_mat2);
}

// Test Case 3: Test with non-square matrices
TEST_F(MmOutTest, NonSquareMatrices) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device_);
  at::Tensor at_mat1 = at::randn({5, 100}, options);
  at::Tensor at_mat2 = at::randn({100, 50}, options);
  run_and_verify(at_mat1, at_mat2);
}

// Test Case 4: Matrix x Vector multiplication
TEST_F(MmOutTest, MatrixVector) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device_);
  at::Tensor at_mat1 = at::randn({50, 20}, options);
  // mat2 is a column vector
  at::Tensor at_mat2 = at::randn({20, 1}, options);
  run_and_verify(at_mat1, at_mat2);
}

// Test Case 5: Vector x Matrix multiplication
TEST_F(MmOutTest, VectorMatrix) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device_);
  // mat1 is a row vector
  at::Tensor at_mat1 = at::randn({1, 50}, options);
  at::Tensor at_mat2 = at::randn({50, 30}, options);
  run_and_verify(at_mat1, at_mat2);
}

#endif // USE_CUDA

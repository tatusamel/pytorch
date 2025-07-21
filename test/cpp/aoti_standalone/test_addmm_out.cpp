#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

// Guard for CUDA specific code
#if defined(USE_CUDA)
#include <torch/csrc/inductor/aoti_standalone/cuda/addmm_out.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

using ::testing::ElementsAreArray;
using torch::standalone::SlimTensor;

// Helper to convert a SlimTensor (potentially on CUDA) to a CPU at::Tensor for easy comparison.
at::Tensor slim_to_cpu_aten(const SlimTensor& slim_tensor) {
    SlimTensor slim_cpu = slim_tensor.to(at::kCPU);
    return at::from_blob(
        slim_cpu.data_ptr(),
        slim_cpu.sizes(),
        slim_cpu.strides(),
        slim_cpu.dtype()
    ).clone(); // Crucial: clone to own the data
}

// Define a test fixture for convenience.
class AddmmOutTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available, skipping test suite.";
        }
        cuda_device_ = at::Device(at::kCUDA);
    }

    // The main verification function that runs both implementations and compares results.
    void run_and_verify(
        const at::Tensor& at_input,
        const at::Tensor& at_mat1,
        const at::Tensor& at_mat2,
        double beta,
        double alpha) {

        // Create SlimTensor views of the at::Tensors
        SlimTensor slim_input = torch::standalone::create_tensor_from_blob(at_input.data_ptr(), at_input.sizes(), at_input.strides(), at_input.scalar_type(), cuda_device_);
        SlimTensor slim_mat1 = torch::standalone::create_tensor_from_blob(at_mat1.data_ptr(), at_mat1.sizes(), at_mat1.strides(), at_mat1.scalar_type(), cuda_device_);
        SlimTensor slim_mat2 = torch::standalone::create_tensor_from_blob(at_mat2.data_ptr(), at_mat2.sizes(), at_mat2.strides(), at_mat2.scalar_type(), cuda_device_);

        // Create output tensors for both implementations. The 'out' tensor for addmm_out
        // must have the same shape as the final result.
        at::Tensor at_out = at::empty({at_mat1.size(0), at_mat2.size(1)}, at_input.options());

        // Create SlimTensor output with explicit shape and contiguous strides
        std::vector<int64_t> expected_shape = {at_mat1.size(0), at_mat2.size(1)};
        std::vector<int64_t> contiguous_strides = torch::standalone::compute_contiguous_strides(expected_shape);
        SlimTensor slim_out = torch::standalone::create_empty_tensor(expected_shape, contiguous_strides, at_out.scalar_type(), cuda_device_);

        AOTITorchError err = aoti_torch_cuda_addmm_out(
            reinterpret_cast<AtenTensorHandle>(&slim_out),
            reinterpret_cast<AtenTensorHandle>(&slim_input),
            reinterpret_cast<AtenTensorHandle>(&slim_mat1),
            reinterpret_cast<AtenTensorHandle>(&slim_mat2),
            beta,
            alpha
        );
        ASSERT_EQ(err, AOTI_TORCH_SUCCESS);


        at::addmm_out(at_out, at_input, at_mat1, at_mat2, beta, alpha);

        // --- Verification ---
        // Convert the slim tensor result to a CPU at::Tensor for comparison
        at::Tensor slim_out_aten_cpu = slim_to_cpu_aten(slim_out);
        at::Tensor at_out_cpu = at_out.to(at::kCPU);

        // Check shapes
        EXPECT_THAT(slim_out.sizes(), ElementsAreArray(at_out.sizes()));

        bool are_close = at::allclose(slim_out_aten_cpu, at_out_cpu, 1e-6, 1e-6);

        ASSERT_TRUE(are_close);
    }

    at::Device cuda_device_ = at::Device(at::kCPU);
};

// Test Case 1: Standard case with beta=1.0, alpha=1.0
TEST_F(AddmmOutTest, StandardFloat) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device_);
    at::Tensor at_input = at::randn({10, 20}, options);
    at::Tensor at_mat1 = at::randn({10, 30}, options);
    at::Tensor at_mat2 = at::randn({30, 20}, options);
    run_and_verify(at_input, at_mat1, at_mat2, 1.0, 1.0);
}

// Test Case 2: Test with different alpha and beta scaling factors
TEST_F(AddmmOutTest, CustomScalarsFloat) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device_);
    at::Tensor at_input = at::randn({15, 25}, options);
    at::Tensor at_mat1 = at::randn({15, 5}, options);
    at::Tensor at_mat2 = at::randn({5, 25}, options);
    run_and_verify(at_input, at_mat1, at_mat2, 0.5, 2.0);
}

// Test Case 3: Test with beta = 0, which should ignore the input tensor
TEST_F(AddmmOutTest, BetaIsZero) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device_);
    // Fill input with NaNs to ensure it's ignored when beta=0
    at::Tensor at_input = at::full({8, 8}, std::numeric_limits<float>::quiet_NaN(), options);
    at::Tensor at_mat1 = at::randn({8, 12}, options);
    at::Tensor at_mat2 = at::randn({12, 8}, options);
    run_and_verify(at_input, at_mat1, at_mat2, 0.0, 1.5);
}

// Test Case 4: Test with half-precision tensors
TEST_F(AddmmOutTest, StandardHalf) {
    auto options = at::TensorOptions().dtype(at::kHalf).device(cuda_device_);
    at::Tensor at_input = at::randn({32, 32}, options);
    at::Tensor at_mat1 = at::randn({32, 64}, options);
    at::Tensor at_mat2 = at::randn({64, 32}, options);
    // Use double for beta/alpha as per the API signature
    run_and_verify(at_input, at_mat1, at_mat2, 1.0, 1.0);
}

// Test Case 5: Test with non-square matrices
TEST_F(AddmmOutTest, NonSquareMatrices) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device_);
    at::Tensor at_input = at::randn({5, 50}, options);
    at::Tensor at_mat1 = at::randn({5, 100}, options);
    at::Tensor at_mat2 = at::randn({100, 50}, options);
    run_and_verify(at_input, at_mat1, at_mat2, 1.2, 0.8);
}

// Test Case 6: add a bias vector to the result of the matrix multiplication
TEST_F(AddmmOutTest, InputBroadcast) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device_);
    // input is a vector of size [25]
    at::Tensor at_input = at::randn({25}, options);
    at::Tensor at_mat1 = at::randn({15, 5}, options);
    at::Tensor at_mat2 = at::randn({5, 25}, options);
    // the output will be [15, 25], so the input vector will be broadcast
    run_and_verify(at_input, at_mat1, at_mat2, 1.0, 1.0);
}

#endif // USE_CUDA

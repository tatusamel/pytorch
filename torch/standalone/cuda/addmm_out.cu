#include <cublas_v2.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>
#include <torch/standalone/cuda/addmm_out.h>

#define CUBLAS_CHECK(condition)                                     \
  do {                                                              \
    cublasStatus_t status = condition;                              \
    TORCH_CHECK(                                                    \
        status == CUBLAS_STATUS_SUCCESS,                            \
        "cuBLAS error: ",                                           \
        cublasGetStatusString(status));                             \
  } while (0)


namespace {

inline std::vector<int64_t> infer_broadcast_strides(
    c10::IntArrayRef original_shape,
    c10::IntArrayRef original_strides,
    c10::IntArrayRef target_shape) {
    const size_t ndim_original = original_shape.size();
    const size_t ndim_target = target_shape.size();
    TORCH_CHECK(ndim_original <= ndim_target, "Cannot broadcast to a smaller number of dimensions");

    std::vector<int64_t> new_strides(ndim_target);
    for (size_t i = 0; i < ndim_target; ++i) {
        const int64_t offset = ndim_target - 1 - i;
        const int64_t original_offset = ndim_original - 1 - i;

        if (original_offset >= 0 && original_shape[original_offset] == target_shape[offset]) {
            new_strides[offset] = original_strides[original_offset];
        } else {
            // This dimension is being broadcast (either it was 1 or it was a new dimension)
            new_strides[offset] = 0;
        }
    }
    return new_strides;
}

template <typename scalar_t>
__global__ void broadcast_copy_kernel(
    scalar_t* out_data,
    const scalar_t* in_data,
    int64_t numel,
    int n_dims,
    const int64_t* out_sizes,
    const int64_t* in_strides) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        int64_t in_offset = 0;
        int64_t remaining_idx = idx;

        // Deconstruct linear index into multi-dim coordinate
        // and calculate input offset using broadcasted strides
        for (int i = n_dims - 1; i >= 0; --i) {
            if (out_sizes[i] == 0) continue;
            int64_t coord = remaining_idx % out_sizes[i];
            remaining_idx /= out_sizes[i];
            in_offset += coord * in_strides[i];
        }
        out_data[idx] = in_data[in_offset];
    }
}

// Host launcher for the broadcast copy kernel
void launch_broadcast_copy(
    torch::standalone::SlimTensor& out,
    const torch::standalone::SlimTensor& in) {

    if (out.numel() == 0) return;
    if (in.numel() == 0) {
        out.zero_();
        return;
    }
    if (out.sizes() == in.sizes()) {
        out.copy_(in); // Use fast path if no broadcasting is needed
        return;
    }

    auto out_shape = out.sizes();
    auto in_b_strides = infer_broadcast_strides(in.sizes(), in.strides(), out_shape);

    const int64_t n_dims = out.dim();
    std::vector<int64_t> out_sizes_vec(out_shape.begin(), out_shape.end());

    int64_t *d_out_sizes, *d_in_strides;
    cudaMalloc(&d_out_sizes, n_dims * sizeof(int64_t));
    cudaMalloc(&d_in_strides, n_dims * sizeof(int64_t));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(d_out_sizes, out_sizes_vec.data(), n_dims * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_in_strides, in_b_strides.data(), n_dims * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    const int64_t block_size = 256;
    const int64_t grid_size = (out.numel() + block_size - 1) / block_size;

    switch(out.dtype()) {
        case c10::kFloat:
            broadcast_copy_kernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<float*>(out.data_ptr()), static_cast<const float*>(in.data_ptr()),
                out.numel(), n_dims, d_out_sizes, d_in_strides);
            break;
        case c10::kDouble:
            broadcast_copy_kernel<double><<<grid_size, block_size, 0, stream>>>(
                static_cast<double*>(out.data_ptr()), static_cast<const double*>(in.data_ptr()),
                out.numel(), n_dims, d_out_sizes, d_in_strides);
            break;
        case c10::kHalf:
            broadcast_copy_kernel<c10::Half><<<grid_size, block_size, 0, stream>>>(
                static_cast<c10::Half*>(out.data_ptr()), static_cast<const c10::Half*>(in.data_ptr()),
                out.numel(), n_dims, d_out_sizes, d_in_strides);
            break;
        default:
            TORCH_CHECK(false, "Unsupported dtype for broadcast copy");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cudaFreeAsync(d_out_sizes, stream);
    cudaFreeAsync(d_in_strides, stream);
}

// This is the core logic. It takes all parameters and calls the correct
// version of cublas<t>gemm.
template <typename scalar_t>
void cublas_gemm_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const scalar_t* alpha,
    const scalar_t* A, int lda,
    const scalar_t* B, int ldb,
    const scalar_t* beta,
    scalar_t* C, int ldc) {
    // This template will fail to compile if called with an unsupported type.
    // We specialize it for the types cuBLAS supports.
    TORCH_CHECK(false, "Unsupported data type for cuBLAS GEMM.");
}

// Specialization for float (SGEMM)
template <>
void cublas_gemm_wrapper<float>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc) {
    CUBLAS_CHECK(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

// Specialization for double (DGEMM)
template <>
void cublas_gemm_wrapper<double>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc) {
    CUBLAS_CHECK(cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

// Specialization for half (HGEMM)
// We will use cublasGemmEx for higher precision accumulation
template <>
void cublas_gemm_wrapper<c10::Half>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const c10::Half* alpha,
    const c10::Half* A, int lda,
    const c10::Half* B, int ldb,
    const c10::Half* beta,
    c10::Half* C, int ldc) {

    float alpha_float = static_cast<float>(*alpha);
    float beta_float = static_cast<float>(*beta);

    // Note: For half precision, alpha and beta must also be half.
    // The cast is safe because we check the scalar types in the calling function.
    CUBLAS_CHECK(cublasGemmEx(
        handle, transa, transb, m, n, k,
        &alpha_float,
        A, CUDA_R_16F, lda,
        B, CUDA_R_16F, ldb,
        &beta_float,
        C, CUDA_R_16F, ldc,
        CUDA_R_32F, // compute in FP32
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}


} // namespace

namespace torch::standalone {

void _cuda_addmm_out(
    const SlimTensor& input,
    const SlimTensor& mat1,
    const SlimTensor& mat2,
    const c10::Scalar& beta,
    const c10::Scalar& alpha,
    SlimTensor& out
) {
    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a 2D tensor");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a 2D tensor");
    TORCH_CHECK(mat1.size(1) == mat2.size(0), "mat1 and mat2 shapes cannot be multiplied");

    // Create a proper vector to hold the expected shape
    std::vector<int64_t> expected_shape = {mat1.size(0), mat2.size(1)};
    c10::IntArrayRef out_shape(expected_shape);

    TORCH_CHECK(mat1.device() == mat2.device() && mat1.device() == out.device(), "All tensors for addmm_out must be on the same device.");
    TORCH_CHECK(out.sizes() == out_shape, "out tensor has incorrect shape");

    c10::cuda::CUDAGuard device_guard(mat1.device());

    launch_broadcast_copy(out, input);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // PyTorch uses row-major layout. cuBLAS expects column-major.
    // To avoid a slow, explicit transpose, we use a mathematical trick:
    // (A @ B)^T = B^T @ A^T
    // We tell cuBLAS to compute `B^T @ A^T` and write it into `C`.
    // Since cuBLAS writes in column-major, the result in memory is equivalent
    // to the row-major representation of `A @ B`.

    // We pass mat2 as the first matrix (A in cuBLAS) and mat1 as the second (B).
    const auto& cublas_matA = mat2;
    const auto& cublas_matB = mat1;
    auto& cublas_C = out;

    // We tell cuBLAS that both input matrices are transposed.
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    // --- Set GEMM Dimensions ---
    // C(m, n) = op(A)(m, k) @ op(B)(k, n)
    // Our operation: out(M, N) = mat1(M, K) @ mat2(K, N)
    // Our trick:     out^T(N, M) = mat2^T(N, K) @ mat1^T(K, M)
    //
    // A -> mat2, B -> mat1, C -> out
    // op(A) -> mat2^T, op(B) -> mat1^T
    // m = rows of op(A) = cols of mat2 = mat2.size(1)
    // n = cols of op(B) = rows of mat1 = mat1.size(0)
    // k = cols of op(A) = rows of mat2 = mat2.size(0)
    int m = mat2.size(1);
    int n = mat1.size(0);
    int k = mat1.size(1);

    // Leading dimensions for row-major matrices are their number of columns.
    // The leading dimension for a column-major matrix is its number of rows.
    // When cuBLAS views our row-major mat2[K,N] as column-major, it sees mat2_T[N,K].
    // The number of rows is N.
    int lda = cublas_matA.size(1);
    int ldb = cublas_matB.size(1);
    int ldc = cublas_C.size(1);

    // --- 6. Dispatch to Correct Kernel ---
    switch (out.dtype()) {
        case c10::kFloat: {
            const float alpha_val = alpha.to<float>();
            const float beta_val = beta.to<float>();
            cublas_gemm_wrapper<float>(handle, transa, transb, m, n, k, &alpha_val,
                static_cast<const float*>(cublas_matA.data_ptr()), lda,
                static_cast<const float*>(cublas_matB.data_ptr()), ldb, &beta_val,
                static_cast<float*>(cublas_C.data_ptr()), ldc);
            break;
        }
        case c10::kDouble: {
            const double alpha_val = alpha.to<double>();
            const double beta_val = beta.to<double>();
            cublas_gemm_wrapper<double>(handle, transa, transb, m, n, k, &alpha_val,
                static_cast<const double*>(cublas_matA.data_ptr()), lda,
                static_cast<const double*>(cublas_matB.data_ptr()), ldb, &beta_val,
                static_cast<double*>(cublas_C.data_ptr()), ldc);
            break;
        }
        case c10::kHalf: {
            const c10::Half alpha_val = alpha.to<c10::Half>();
            const c10::Half beta_val = beta.to<c10::Half>();
            cublas_gemm_wrapper<c10::Half>(handle, transa, transb, m, n, k, &alpha_val,
                static_cast<const c10::Half*>(cublas_matA.data_ptr()), lda,
                static_cast<const c10::Half*>(cublas_matB.data_ptr()), ldb, &beta_val,
                static_cast<c10::Half*>(cublas_C.data_ptr()), ldc);
            break;
        }
        default:
            TORCH_CHECK(false, "addmm_out: Unsupported data type");
    }

    CUBLAS_CHECK(cublasDestroy(handle));
}

} // namespace torch::standalone

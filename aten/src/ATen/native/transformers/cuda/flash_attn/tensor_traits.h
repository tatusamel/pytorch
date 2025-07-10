#pragma once
#include <ATen/ops/all_ops.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/standalone/pad_template.h>
#include <torch/standalone/reshape_template.h>



namespace FLASH_NAMESPACE {

template <typename T>
struct TensorTraits;

// Specialization for at::Tensor

template <>
struct TensorTraits<at::Tensor> {

    using TensorType = at::Tensor;
    using GeneratorType = at::Generator;

    static TensorType reshape(const TensorType& t, c10::IntArrayRef shape) { return at::reshape(t, shape); }
    static TensorType pad(const TensorType& t, c10::IntArrayRef pad_dims, const std::string& mode = "constant", double value = 0.0) { return at::pad(t, pad_dims, mode, value); }
    static TensorType empty_like(const TensorType& t) { return at::empty_like(t); }
    static TensorType empty(c10::IntArrayRef size, const c10::TensorOptions& opts) { return at::empty(size, opts);}
    static TensorType zeros(c10::IntArrayRef size, const c10::TensorOptions& opts) { return at::zeros(size, opts); }
    static TensorType& sum_out(TensorType& out, const TensorType& t, c10::IntArrayRef dim, bool keepdim = false, std::optional<c10::ScalarType> dtype = std::nullopt) {
        return at::sum_out(out, t, dim, keepdim, dtype);
    }

    static cudaDeviceProp* getCurrentDeviceProperties() {
        return at::cuda::getCurrentDeviceProperties();
    }
};

// Specialization for torch::standalone::SlimTensor

template <>
struct TensorTraits<torch::standalone::SlimTensor> {
    using TensorType = torch::standalone::SlimTensor;
    using GeneratorType = at::Generator; // c10::intrusive_ptr<c10::GeneratorImpl>;

    static TensorType zeros(c10::IntArrayRef size, const c10::TensorOptions& opts) {
        return torch::standalone::zeros(size, opts);
    }
    static TensorType reshape(const TensorType& t, c10::IntArrayRef shape) {
        return torch::standalone::_reshape(t, shape);
    }

    static TensorType pad(const TensorType& t, c10::IntArrayRef pad_dims, const std::string& mode = "constant", double value = 0.0) {
        return torch::standalone::_pad(t, pad_dims, mode, std::optional<double>(value));
    }

    static TensorType empty_like(const TensorType& t) {
        return torch::standalone::empty_like(t);
    }

    // todo: implement a shim version of sum_out
    static at::Tensor& sum_out(at::Tensor& out, const at::Tensor& t, c10::IntArrayRef dim, bool keepdim = false, std::optional<c10::ScalarType> dtype = std::nullopt) {
        TORCH_CHECK(false, "shim version of sum_out is not implemented yet");
    }

    static TensorType empty(c10::IntArrayRef size, const c10::TensorOptions& opts) {
        return torch::standalone::create_empty_tensor(
            size,
            torch::standalone::compute_contiguous_strides(size),
            opts.dtype().toScalarType(),
            opts.device(),
            0
        );
    }

    static cudaDeviceProp* getCurrentDeviceProperties() {
        static thread_local cudaDeviceProp prop;
        static thread_local bool initialized = false;

        if (!initialized) {
            int device;
            cudaGetDevice(&device);
            cudaGetDeviceProperties(&prop, device);
            initialized = true;
        }
        return &prop;
    }
};

} // namespace FLASH_NAMESPACE

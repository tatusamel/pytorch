#pragma once

#include <c10/core/Scalar.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

namespace torch::standalone {

// CUDA implementation of addmm_out
void _cuda_addmm_out(
    const SlimTensor& input,
    const SlimTensor& mat1,
    const SlimTensor& mat2,
    const c10::Scalar& beta,
    const c10::Scalar& alpha,
    SlimTensor& out);

} // namespace torch::standalone

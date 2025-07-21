#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/cuda/addmm_out.h>

#ifdef __cplusplus
extern "C" {
#endif

using torch::standalone::SlimTensor;

AOTITorchError aoti_torch_cuda_addmm_out(
    AtenTensorHandle out_handle,
    AtenTensorHandle self_handle,
    AtenTensorHandle mat1_handle,
    AtenTensorHandle mat2_handle,
    double beta,
    double alpha) {
    SlimTensor* self = reinterpret_cast<SlimTensor*>(self_handle);
    SlimTensor* mat1 = reinterpret_cast<SlimTensor*>(mat1_handle);
    SlimTensor* mat2 = reinterpret_cast<SlimTensor*>(mat2_handle);
    SlimTensor* out = reinterpret_cast<SlimTensor*>(out_handle);

    torch::standalone::_cuda_addmm_out(*self, *mat1, *mat2, beta, alpha, *out);
    return AOTI_TORCH_SUCCESS;

}

#ifdef __cplusplus
} // extern "C"
#endif

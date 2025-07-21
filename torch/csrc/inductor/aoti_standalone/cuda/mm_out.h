#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/cuda/addmm_out.h>

#ifdef __cplusplus
extern "C" {
#endif

using torch::standalone::SlimTensor;

AOTITorchError aoti_torch_cuda_mm_out(
    AtenTensorHandle out_handle,
    AtenTensorHandle self_handle,
    AtenTensorHandle mat2_handle) {
  SlimTensor* self = reinterpret_cast<SlimTensor*>(self_handle);
  SlimTensor* mat2 = reinterpret_cast<SlimTensor*>(mat2_handle);
  SlimTensor* out = reinterpret_cast<SlimTensor*>(out_handle);

  auto dummy_input = torch::standalone::create_empty_tensor(
      out->sizes(), out->strides(), out->dtype());

  torch::standalone::_cuda_addmm_out(dummy_input, *self, *mat2, 0.0, 1.0, *out);
  return AOTI_TORCH_SUCCESS;
}

#ifdef __cplusplus
} // extern "C"
#endif

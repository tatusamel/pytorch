#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/standalone/transpose_int_template.h>

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda_transpose_int(
    AtenTensorHandle self,
    int64_t dim0,
    int64_t dim1,
    AtenTensorHandle* ret0) {
  torch::standalone::SlimTensor* self_tensor =
      reinterpret_cast<torch::standalone::SlimTensor*>(self);
  torch::standalone::SlimTensor result_tensor =
      transpose_template(*self_tensor, dim0, dim1);
  *ret0 = reinterpret_cast<AtenTensorHandle>(
      new torch::standalone::SlimTensor(std::move(result_tensor)));
  return AOTI_TORCH_SUCCESS;
}

#ifdef __cplusplus
} // extern "C"
#endif

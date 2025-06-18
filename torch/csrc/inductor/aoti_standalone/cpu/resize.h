#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/resize_template.h>

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cpu_resize_(
    AtenTensorHandle self,
    const int64_t* size,
    int64_t size_len_,
    int32_t* memory_format) {
<<<<<<< HEAD
  torch::standalone::SlimTensor* tensor =
      reinterpret_cast<torch::standalone::SlimTensor*>(self);
  c10::IntArrayRef size_ref(size, size_len_);
  std::optional<c10::MemoryFormat> optional_memory_format;
  if (memory_format) {
    optional_memory_format = static_cast<c10::MemoryFormat>(*memory_format);
=======
  try {
    SlimTensor* tensor = reinterpret_cast<SlimTensor*>(self);
    ArrayRef size_ref(size, size_len_);
    std::optional<c10::MemoryFormat> optional_memory_format;
    if (memory_format) {
      optional_memory_format = static_cast<c10::MemoryFormat>(*memory_format);
    }
    _resize_(*tensor, size_ref, optional_memory_format);
    return AOTI_TORCH_SUCCESS;
  } catch (const std::exception& e) {
    return AOTI_TORCH_FAILURE;
>>>>>>> 8089a961245 (change structure of resize headers)
  }
  torch::standalone::_resize_(*tensor, size_ref, optional_memory_format);
  return AOTI_TORCH_SUCCESS;
}

#ifdef __cplusplus
} // extern "C"
#endif

#pragma once
#include <c10/core/MemoryFormat.h>
#include <torch/standalone/resize_template.h>
#include <torch/standalone/slim_tensor/array_ref.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/standalone/slim_tensor/storage.h>

namespace torch::standalone {

inline const SlimTensor& resize_(
    AtenTensorHandle self,
    const int64_t* size,
    int64_t size_len_,
    int32_t* memory_format) {
  SlimTensor* tensor = reinterpret_cast<SlimTensor*>(self);
  ArrayRef size_ref(size, size_len_);
  std::optional<c10::MemoryFormat> optional_memory_format;
  if (memory_format) {
    optional_memory_format = static_cast<c10::MemoryFormat>(*memory_format);
  }
  return _resize_(*tensor, size_ref, optional_memory_format);
}

} // namespace torch::standalone

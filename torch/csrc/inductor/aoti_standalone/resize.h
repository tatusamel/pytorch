#pragma once
#include <c10/core/MemoryFormat.h>
#include <torch/standalone/resize_template.h>
#include <torch/standalone/slim_tensor/array_ref.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/standalone/slim_tensor/storage.h>

namespace torch::standalone {

inline const SlimTensor& resize_(
    const SlimTensor& self,
    ArrayRef size,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  return _resize_<SlimTensor, ArrayRef>(self, size, optional_memory_format);
}

} // namespace torch::standalone

#pragma once
#include <torch/csrc/inductor/aoti_standalone/utils.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

namespace torch::standalone {

template <class T, class AREF>
inline void resize_template(
  T& tensor,
  const int64_t* new_sizes,
  int64_t new_len,
  std::optional<c10::MemoryFormat> optional_memory_format
) {

if ()

}

} // namespace torch::standalone

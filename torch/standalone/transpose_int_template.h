#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/csrc/inductor/aoti_standalone/utils.h>
#include <algorithm>
#include <vector>

namespace torch::standalone {

template <typename T>
inline T transpose_template(const T& self, int64_t dim0, int64_t dim1) {
  int64_t ndim = self.dim();
  dim0 = torch::standalone::maybe_wrap_dim(dim0, ndim);
  dim1 = torch::standalone::maybe_wrap_dim(dim1, ndim);

  // if dimensions are the same, return a copy that is an alias.
  // TODO: check if true
  if (dim0 == dim1) {
    return self;
  }

  std::vector<int64_t> new_sizes(self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> new_strides(
      self.strides().begin(), self.strides().end());

  // swap the metadata
  std::swap(new_sizes[dim0], new_sizes[dim1]);
  std::swap(new_strides[dim0], new_strides[dim1]);

  // Create a new tensor that is a view of the original
  // It shares the new storage but has the new sizes and strides
  T result = self;
  result.as_strided_(
      c10::IntArrayRef(new_sizes),
      c10::IntArrayRef(new_strides),
      self.storage_offset());
  return result;
}

} // namespace torch::standalone

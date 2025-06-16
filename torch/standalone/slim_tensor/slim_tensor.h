#pragma once

#include <c10/core/MemoryFormat.h>
#include <torch/standalone/slim_tensor/storage.h>
#include <torch/standalone/slim_tensor/utils.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <c10/core/MemoryFormat.h>
#include <c10/core/impl/SizesAndStrides.h>
#include <torch/csrc/inductor/aoti_standalone/utils.h>
#include <c10/core/MemoryFormat.h>
#include <torch/standalone/slim_tensor/storage.h>
#include <torch/standalone/slim_tensor/utils.h>

namespace torch::standalone {

class SlimTensor {
 public:
  SlimTensor(
      Storage&& storage,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      c10::ScalarType dtype,
      int64_t storage_offset = 0)
      : storage_(std::move(storage)),
        storage_offset_(storage_offset),
        numel_(torch::standalone::compute_numel(sizes)),
        dtype_(dtype) {
    set_sizes_and_strides(sizes, strides);
  }

  SlimTensor() = delete;
  SlimTensor(const SlimTensor&) = default;
  SlimTensor& operator=(const SlimTensor&) = default;
  SlimTensor(SlimTensor&&) = default;
  SlimTensor& operator=(SlimTensor&&) = default;

  ~SlimTensor() = default;

  void reset() {
    // Decrement the refcount of the storage
    storage_.reset();
  }

  // Accessors
  Storage storage() const {
    return storage_;
  }

  c10::IntArrayRef sizes() const {
    return sizes_and_strides_.sizes_arrayref();
  }

  int64_t size(size_t dim) const {
    return sizes_and_strides_.size_at(dim);
  }

  c10::IntArrayRef strides() const {
    return sizes_and_strides_.strides_arrayref();
  }

  int64_t stride(size_t dim) const {
    return sizes_and_strides_.stride_at(dim);
  }

  c10::ScalarType dtype() const {
    return dtype_;
  }

  const c10::Device& device() const {
    return storage_->device();
  }

  c10::DeviceType device_type() const {
    return storage_->device_type();
  }

  c10::DeviceIndex device_index() const {
    return storage_->device_index();
  }

  int64_t storage_offset() const {
    return storage_offset_;
  }

  size_t numel() const {
    return numel_;
  }

  size_t nbytes() const {
    return compute_nbytes(numel_, dtype_);
  }

  size_t dim() const {
    return sizes_and_strides_.size();
  }

  void* data_ptr() const {
    return storage_->data();
  }

  bool is_contiguous() const {
    return is_contiguous_;
  }

  void set_storage(Storage&& new_storage) {
    storage_ = std::move(new_storage);
  }

  void set_sizes_and_strides(
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      std::optional<int64_t> storage_offset = std::nullopt) {
<<<<<<< HEAD
    const int64_t new_dim = static_cast<int64_t>(sizes.size());
    TORCH_CHECK(
        new_dim == static_cast<int64_t>(strides.size()),
=======
    const int64_t new_dim = static_cast<int64_t>(new_sizes.size());
    TORCH_CHECK(
        new_dim == static_cast<int64_t>(new_strides.size()),
>>>>>>> 1ed5e4e19ca (fix compiler warnings as they might lead to infinite loop)
        "dimensionality of sizes (",
        new_dim,
        ") must match dimensionality of strides (",
        strides.size(),
        ")");

    std::vector<int64_t> new_sizes = sizes.vec();
    std::vector<int64_t> new_strides = strides.vec();

    // stride calculation logic
    bool overflowed = false;
    if (new_dim > 0) {
      for (int64_t dim = new_dim - 1; dim >= 0; dim--) {
<<<<<<< HEAD
        if (strides[dim] >= 0) {
          new_strides[dim] = strides[dim];
=======
        if (new_strides[dim] >= 0) {
          new_strides_data[dim] = new_strides[dim];
>>>>>>> 1ed5e4e19ca (fix compiler warnings as they might lead to infinite loop)
        } else {
          // for negative strides
          if (dim == new_dim - 1) {
            new_strides[dim] = 1;
          } else {
            overflowed |= c10::mul_overflows(
                new_strides[dim + 1],
                std::max<int64_t>(new_sizes[dim + 1], 1),
                &new_strides[dim]);
          }
        }
      }
    }
    TORCH_CHECK(!overflowed, "Stride calculation overflowed");

    sizes_and_strides_.set_sizes(new_sizes);
    sizes_and_strides_.set_strides(new_strides);
    if (storage_offset.has_value()) {
      storage_offset_ = *storage_offset;
    }

    refresh_numel();
    refresh_contiguous();
  }

  void set_sizes_contiguous(c10::IntArrayRef sizes) {
    sizes_and_strides_.set_sizes(sizes);
    refresh_numel();
    this->empty_tensor_restride(c10::MemoryFormat::Contiguous);
  }

  void empty_tensor_restride(c10::MemoryFormat memory_format) {
    switch (memory_format) {
      case c10::MemoryFormat::Contiguous: {
        const auto current_dim = this->dim();
        std::vector<int64_t> new_strides_data(current_dim);

        if (current_dim > 0) {
          const int64_t last_idx = static_cast<int64_t>(current_dim) - 1;
          new_strides_data[last_idx] = 1;
          for (int64_t i = last_idx - 1; i >= 0; i--) {
            new_strides_data[i] = new_strides_data[i + 1] * this->size(i + 1);
          }
        }
        sizes_and_strides_.set_strides(new_strides_data);
        break;
      }
      // TODO: implement the other cases.
      case c10::MemoryFormat::ChannelsLast:
      case c10::MemoryFormat::Preserve:
      case c10::MemoryFormat::ChannelsLast3d:
      default:
        TORCH_CHECK(false, "Only support MemoryFormat::Contiguous now");
        break;
    }

    refresh_contiguous();
  }

  SlimTensor as_strided_(
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      int64_t storage_offset) {
    sizes_and_strides_.set_sizes(sizes);
    sizes_and_strides_.set_strides(strides);
    storage_offset_ = storage_offset;
    return *this;
  }

  SlimTensor copy_(const SlimTensor& other) {
    storage_->clone(other.storage(), other.nbytes(), other.storage_offset());
    return *this;
  }

  SlimTensor to(const c10::Device& device) const {
    // Does not mutate the current tensor, but returns a new tensor
    if (device == storage_->device()) {
      return *this;
    }
    Storage new_storage(new MaybeOwningStorage(nbytes(), device));
    new_storage->clone(storage_, nbytes(), storage_offset_);
    return SlimTensor(
        std::move(new_storage),
        sizes_and_strides_.sizes_arrayref(),
        sizes_and_strides_.strides_arrayref(),
        dtype_,
        storage_offset_);
  }

  SlimTensor to(c10::ScalarType dtype) const {
    throw std::runtime_error("TBD: to(dtype)");
  }

  SlimTensor permute(c10::IntArrayRef dims) const {
    const size_t ndim = this->dim();

    TORCH_CHECK(
        ndim == static_cast<size_t>(dims.size()),
        "permute: dims length must be equal to tensor.dim()")

    const auto old_sizes = this->sizes();
    const auto old_strides = this->strides();

    std::vector<int64_t> new_sizes(ndim);
    std::vector<int64_t> new_strides(ndim);
    std::vector<bool> seen_dims(ndim, false);

    for (size_t i = 0; i < ndim; i++) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      int64_t d = torch::standalone::maybe_wrap_dim(dims[i], ndim);
      TORCH_CHECK(!seen_dims[d], "permute: duplicate dims are not allowed");
      seen_dims[d] = true;
      new_sizes[i] = old_sizes[d];
      new_strides[i] = old_strides[d];
    }

    SlimTensor result = *this;
    result.as_strided_(new_sizes, new_strides, this->storage_offset());
    return result;
  }

 private:
  void refresh_numel() {
    numel_ =
        torch::standalone::compute_numel(sizes_and_strides_.sizes_arrayref());
  }

  bool compute_is_contiguous() const {
    if (dim() <= 1) {
      return true;
    }

    int64_t expected_stride = 1;
    for (int64_t i = static_cast<int64_t>(dim()) - 1; i >= 0; i--) {
      if (size(i) == 0) {
        return true;
      }
      if (size(i) != 1 && stride(i) != expected_stride) {
        return false;
      }
      expected_stride *= size(i);
    }
    return true;
  }

  void refresh_contiguous() {
    // In SlimTensor, we only care about the single is_contiguous_ flag.
    // (because TensorImpl (aten) implementation has other stuff)
    is_contiguous_ = compute_is_contiguous();
  }

  Storage storage_; // device_type_ and device_index_ are stored in storage_
  int64_t storage_offset_;
  c10::impl::SizesAndStrides sizes_and_strides_;
  // If sizes and strides are empty, the numel is 1!!  However, most of the
  // time, we will immediately set sizes to {0} and reset numel to 0.
  // (Can't do that in the default initializers, because there's no way to
  // spell "allocate a one-element array" for strides_).
  size_t numel_ = 1;
  c10::ScalarType dtype_;
  bool is_contiguous_ = true;
  // NOLINTNEXTLINE(clang-diagnostic-unused-private-field)
  std::array<int8_t, 6> reserved_{}; // padding to align to 8 bytes
};

// The returned SlimTensor owns the underlying storage
inline SlimTensor create_empty_tensor(
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0,
    bool own_sizes_and_strides = false) {
  size_t nbytes = compute_nbytes(sizes, dtype);
  Storage storage(new MaybeOwningStorage(nbytes, device));
  return SlimTensor(std::move(storage), sizes, strides, dtype, storage_offset);
}

// The returned SlimTensor does not own the underlying storage
inline SlimTensor create_tensor_from_blob(
    void* data,
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0) {
  if (data == nullptr) {
    throw std::runtime_error("data pointer can not be nullptr");
  }
  Storage storage(new MaybeOwningStorage(data, device));
  return SlimTensor(std::move(storage), sizes, strides, dtype, storage_offset);
}
} // namespace torch::standalone

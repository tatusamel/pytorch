#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>

#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/impl/SizesAndStrides.h>
#include <torch/csrc/inductor/aoti_standalone/utils.h>
#include <torch/standalone/reshape_template.h>
#include <torch/standalone/slim_tensor/storage.h>
#include <torch/standalone/slim_tensor/utils.h>
#include <torch/standalone/transpose_int_template.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

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

  int64_t size(int64_t dim) const {
    return sizes_and_strides_.size_at(torch::standalone::maybe_wrap_dim(
        dim, static_cast<int64_t>(this->dim())));
  }

  c10::IntArrayRef strides() const {
    return sizes_and_strides_.strides_arrayref();
  }

  int64_t stride(int64_t dim) const {
    return sizes_and_strides_.stride_at(torch::standalone::maybe_wrap_dim(
        dim, static_cast<int64_t>(this->dim())));
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
    const int64_t new_dim = static_cast<int64_t>(sizes.size());
    TORCH_CHECK(
        new_dim == static_cast<int64_t>(strides.size()),
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
        if (strides[dim] >= 0) {
          new_strides[dim] = strides[dim];
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
    set_sizes_and_strides(sizes, strides);
    storage_offset_ = storage_offset;

    refresh_numel();
    refresh_contiguous();
    return *this;
  }

  SlimTensor copy_(const SlimTensor& other) {
    TORCH_CHECK(
        this->numel() == other.numel(), "copy_: numel of tensors must match");

    if (this->numel() == 0) {
      return *this;
    }

    // Case 1: Both tensors are contiguous. We can do a fast bulk copy.
    if (this->is_contiguous() && other.is_contiguous()) {
      int64_t src_byte_offset = other.storage_offset() *
          static_cast<int64_t>(c10::elementSize(this->dtype_));
      storage_->clone(other.storage(), other.nbytes(), src_byte_offset);
      return *this;
    }

    // Case 2: At least one tensor is non-contiguous, perform element-wise copy
    // that respects both source and destination strides.
    const size_t elem_size = c10::elementSize(dtype_);
    char* dst_data = static_cast<char*>(this->data_ptr()) +
        this->storage_offset() * elem_size;
    const char* src_data = static_cast<const char*>(other.data_ptr()) +
        other.storage_offset() * elem_size;

    std::vector<int64_t> counter(this->dim(), 0);
    for (size_t i = 0; i < this->numel(); i++) {
      // Compute src offset in elements
      int64_t src_offset = 0;
      for (size_t d = 0; d < other.dim(); d++) {
        src_offset += counter[d] * other.stride(d);
      }

      // Compute dst offset in elements
      int64_t dst_offset = 0;
      for (size_t d = 0; d < this->dim(); d++) {
        dst_offset += counter[d] * this->stride(d);
      }

      // Copy elem_size bytes from src to dst
      if (this->device().is_cpu() && other.device().is_cpu()) {
        std::memcpy(
            dst_data + dst_offset * elem_size,
            src_data + src_offset * elem_size,
            elem_size);
      } else if (this->device().is_cuda() || other.device().is_cuda()) {
#if defined(USE_CUDA)
        DeviceTraits<c10::DeviceType::CUDA>::memcpy(
            dst_data + dst_offset * elem_size,
            src_data + src_offset * elem_size,
            elem_size,
            device(), // dst device
            other.device() // src device
        );
#else
        TORCH_CHECK(false, "copy_: no CUDA support");
#endif
      }
      // Increment the multi-dimensional counter
      for (int64_t d = static_cast<int64_t>(this->dim()) - 1; d >= 0; --d) {
        counter[d]++;
        if (counter[d] < this->size(d)) {
          break;
        }
        counter[d] = 0;
      }
    }
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

  void fill_(const c10::Scalar& value) {
    auto fill_value = [&](auto typed_value) {
      using SType = decltype(typed_value);
      if (this->device().is_cuda()) {
#ifdef USE_CUDA
        if constexpr (std::is_same_v<SType, bool>) {
          // Special handling for bool since std::vector<bool> doesn't have
          // data()
          std::vector<uint8_t> host_data(this->numel(), typed_value ? 1 : 0);
          cudaError_t err = cudaMemcpy(
              this->data_ptr(),
              host_data.data(),
              this->nbytes(),
              cudaMemcpyHostToDevice);
          TORCH_CHECK(
              err == cudaSuccess,
              "CUDA memcpy failed: ",
              cudaGetErrorString(err));
        } else {
          std::vector<SType> host_data(this->numel(), typed_value);
          cudaError_t err = cudaMemcpy(
              this->data_ptr(),
              host_data.data(),
              this->nbytes(),
              cudaMemcpyHostToDevice);
          TORCH_CHECK(
              err == cudaSuccess,
              "CUDA memcpy failed: ",
              cudaGetErrorString(err));
        }
#else
        TORCH_CHECK(false, "CUDA support not available");
#endif
      } else if (this->device().is_cpu()) {
        // Fill all elements, not just the first one
        SType* data = static_cast<SType*>(this->data_ptr());
        for (size_t i = 0; i < this->numel(); ++i) {
          data[i] = typed_value;
        }
      }
    };

    switch (this->dtype()) {
      case c10::ScalarType::Double:
        fill_value(value.to<double>());
        break;
      case c10::ScalarType::Float:
        fill_value(value.to<float>());
        break;
      case c10::ScalarType::Long:
        fill_value(value.to<int64_t>());
        break;
      case c10::ScalarType::Int:
        fill_value(value.to<int32_t>());
        break;
      case c10::ScalarType::Short:
        fill_value(value.to<int16_t>());
        break;
      case c10::ScalarType::Char:
        fill_value(value.to<int8_t>());
        break;
      case c10::ScalarType::Byte:
        fill_value(value.to<uint8_t>());
        break;
      case c10::ScalarType::Bool:
        fill_value(value.to<bool>());
        break;
      default:
        TORCH_CHECK(false, "fill_: Unsupported dtype");
    }
  }
  SlimTensor transpose(int64_t dim0, int64_t dim1) const {
    return _transpose(*this, dim0, dim1);
  }

  // returns a new tensor that is a narrowed version of this and storage is
  // shared
  SlimTensor narrow(int64_t dim, int64_t start, int64_t length) const {
    TORCH_CHECK(
        this->dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");

    dim = torch::standalone::maybe_wrap_dim(
        dim, static_cast<int64_t>(this->dim()));
    int64_t end = start + length;

    TORCH_CHECK(length >= 0, "narrow(): length must be non-negative.");
    TORCH_CHECK(
        dim >= 0 && dim < static_cast<int64_t>(this->dim()),
        "Dimension out of range");
    TORCH_CHECK(
        start >= 0 && end <= this->size(dim),
        "Invalid range to narrow. range(start, start+length) must be a subset of range(0, )",
        this->size(dim),
        ").");

    SlimTensor result = *this;

    int64_t new_storage_offset =
        this->storage_offset() + start * this->stride(dim);

    std::vector<int64_t> new_sizes = this->sizes().vec();
    new_sizes[dim] = length;

    result.as_strided_(new_sizes, this->strides(), new_storage_offset);
    return result;
  }

  SlimTensor& reshape_as_view(c10::IntArrayRef new_shape) {
    this->set_sizes_contiguous(new_shape);
    // after a clone, the new view is relative to the start of the storage.
    this->storage_offset_ = 0;
    return *this;
  }

  SlimTensor reshape(c10::IntArrayRef proposed_shape) const {
    return _reshape(*this, proposed_shape);
  }

  SlimTensor& zero_() {
    fill_(c10::Scalar(0));
    return *this;
  }

  SlimTensor clone_contiguous() const;

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

inline SlimTensor SlimTensor::clone_contiguous() const {
  std::vector<int64_t> contig_strides =
      torch::standalone::compute_contiguous_strides(this->sizes());

  SlimTensor result = create_empty_tensor(
      this->sizes(),
      c10::IntArrayRef(contig_strides),
      this->dtype(),
      this->device(),
      0);
  // copy the data from (potentially non-contiguous) the self tensor
  result.copy_(*this);
  return result;
}
inline SlimTensor zeros(c10::IntArrayRef size, const c10::TensorOptions& opts) {
  SlimTensor result = create_empty_tensor(
      size,
      compute_contiguous_strides(size),
      opts.dtype().toScalarType(),
      opts.device(),
      0);
  result.zero_();
  return result;
}

inline SlimTensor empty_like(const SlimTensor& other) {
  return create_empty_tensor(
      other.sizes(), other.strides(), other.dtype(), other.device(), 0);
}

} // namespace torch::standalone

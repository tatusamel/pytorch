#pragma once
#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/standalone/slim_tensor/array_ref.h>
#include <torch/standalone/cpu/resize_template.h>
#include <torch/standalone/slim_tensor/storage.h>

#ifdef __cplusplus
extern "C" {
#endif

namespace torch::standalone {

  // resize_bytes_cpu in ATen/native/Resize.cpp
  inline void resize_bytes_cpu(MaybeOwningStorage* storage, size_t new_size_bytes) {
    TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");

    void* new_data = nullptr;
    const c10::Device& device = storage->device();

    if (new_size_bytes > 0) {
      new_data = DeviceTraits<c10::DeviceType::CPU>::allocate(new_size_bytes);
    }

    void* old_data = storage->data();
    const size_t old_capacity = storage->nbytes();
    const size_t copy_capacity = std::min(new_size_bytes, old_capacity);

    if (old_data != nullptr && copy_capacity > 0 ) {
      DeviceTraits<c10::DeviceType::CPU>::memcpy(new_data, old_data, copy_capacity, device, device);
    }

    if (old_data != nullptr) {
      DeviceTraits<c10::DeviceType::CPU>::free(old_data);
    }

    storage->set_data_ptr(new_data);
    storage->set_nbytes(new_size_bytes);
  }

  inline int64_t computeStorageNbytesContiguous(
    ArrayRef<int64_t> sizes,
    size_t itemsize,
    int64_t storage_offset) {

      int64_t numel = 1;
      for (auto s : sizes) {
        numel *= s;
      }
      return itemsize * (storage_offset + numel);
    }


  inline int64_t computeStorageNbytes(
    ArrayRef<int64_t> sizes,
    ArrayRef<int64_t> strides,
    size_t itemsize,
    int64_t storage_offset) {

      if (sizes.empty()) {
        return itemsize * storage_offset;
      }

      int64_t size = 1;
      for (size_t i = 0; i < sizes.size(); i++) {
        if (sizes[i] == 0) {
          return 0;
        }
        size += strides[i] * (sizes[i] - 1);
      }
      return itemsize * (storage_offset + size);

  }

  inline void _maybe_resize_storage_cpu(SlimTensor* self, int64_t new_size_bytes) {
    if (self->numel() == 0) {
      return;
    }

    const Storage& storage = self->storage();

    if (!storage) {
      Storage new_storage = new MaybeOwningStorage(new_size, self->device());
      self->set_storage(std::move(new_storage));
    } else if (new_size_bytes > storage->nbytes()){
      resize_bytes_cpu(storage.get(), new_size_bytes)
    }
  }


  inline SlimTensor* _resize_impl_(
    SlimTensor* self,
    ArrayRef<int64_t> size,
    std::optional<ArrayRef<int64_t>> stride,
    bool resize_storage) {

      if (self->sizes() == size && (!stride || self->strides() == stride.value())) {
        return self;
      }

      const auto itemsize = c10::elementSize(self->dtype());
      const auto storage_offset = self->storage_offset();
      int64_t storage_size = 1;

      if (stride) {
        self->set_sizes_and_strides(size, *stride);
        storage_size = computeStorageNbytes(size, *stride, itemsize, storage_offset);
      } else {
        self->set_sizes_contiguous(size);
        storage_size = computeStorageNbytesContiguous(size, itemsize, storage_offset);
      }

      if (resize_storage) {
        _maybe_resize_storage_cpu(self, storage_size);
      }

      return self;
    }


  inline const SlimTensor& _resize_ (
    const SlimTensor& self,
    ArrayRef<int64_t> size,
    std::optional<c10::MemoryFormat> optional_memory_format) {

      SlimTensor* self_ = const_cast<SlimTensor*>(&self);
      _resize_impl_(self_, size, /*stride=*/std::nullopt, true);

      if (optional_memory_format.has_value()) {
        c10::MemoryFormat memory_format = static_cast<c10::MemoryFormat>(optional_memory_format.value());
        AOTI_TORCH_CHECK(memory_format != c10::MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
        self_->empty_tensor_restride(memory_format);
      }
      // TODO: always do enabling deterministic operations thing here without if check.
      return self;
  }

  AOTITorchError aoti_torch_cpu_resize_(AtenTensorHandle self, const int64_t* size, int64_t size_len_, int32_t* memory_format) {

    SlimTensor* tensor = reinterpret_cast<SlimTensor*>(self);
    ArrayRef size_ref(size, size_len_);
    std::optional<c10::MemoryFormat> optional_memory_format;
    if (memory_format) {
      optional_memory_format = static_cast<c10::MemoryFormat>(*memory_format);
    }
    _resize_(*tensor, size_ref, optional_memory_format);

    // resize_template<SlimTensor, ArrayRef>(tensor, size, size_len, optional_memory_format);

    return AOTI_TORCH_SUCCESS;

  }

}


#ifdef __cplusplus
} // extern "C"
#endif

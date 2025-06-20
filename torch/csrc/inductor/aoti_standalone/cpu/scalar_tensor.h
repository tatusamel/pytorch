#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/scalar_tensor_template.h>

namespace torch::standalone {
AOTITorchError aoti_torch_cpu_scalar_tensor(double s, int32_t* dtype, int32_t* layout, int32_t* device, int32_t device_index_, int32_t* pin_memory, AtenTensorHandle* ret0) {
  try{
    c10::Scalar s_scalar(s);
    c10::ScalarType dtype_val = static_cast<c10::ScalarType>(*dtype);

    TORCH_CHECK(*device == static_cast<int32_t>(c10::DeviceType::CPU), "Device must be CPU");
    c10::Device device_val(c10::DeviceType::CPU, device_index_);

    SlimTensor tensor = scalar_tensor_template<SlimTensor, ArrayRef>(s_scalar, dtype_val, device_val);

    *ret0 = reinterpret_cast<AtenTensorHandle>(new SlimTensor(std::move(tensor)));
    return AOTI_TORCH_SUCCESS;
  } catch (const std::exception& e) {
    return AOTI_TORCH_FAILURE;
  }
}
} // namespace torch::standalone

#pragma once

#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/slim_tensor/utils.h>
#include <torch/standalone/util/Factory.h>


namespace torch::standalone {

template <typename T>
inline void fill_scalar(T& tensor, const c10::Scalar& value) {
  if (tensor.numel() != 1) {
    TORCH_CHECK(false, "fill_scalar is only for tensors with 1 element");
  }

  auto fill_value = [&](auto typed_value) {
    using SType = decltype(typed_value);
    *static_cast<SType*>(tensor.data_ptr()) = typed_value;
  };

  switch (tensor.dtype()) {
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
      TORCH_CHECK(false, "Unsupported dtype in fill_scalar");
  }
}

template <class T, class AREF>
T scalar_tensor_template(
  const c10::Scalar& s,
  c10::ScalarType dtype,
  c10::Device device) {

  AREF sizes = {};
  AREF strides = {};

  T result = empty_tensor<T, AREF>(sizes, strides, dtype, device);

  fill_scalar(result, s);
  return result;

}

} // namespace torch::standalone

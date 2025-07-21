#pragma once

#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/torch.h>

// Helper to convert a SlimTensor (potentially on CUDA) to a CPU at::Tensor for
// easy comparison.
inline at::Tensor slim_to_cpu_aten(
    const torch::standalone::SlimTensor& slim_tensor) {
  torch::standalone::SlimTensor slim_cpu = slim_tensor.to(at::kCPU);
  return at::from_blob(
             slim_cpu.data_ptr(),
             slim_cpu.sizes(),
             slim_cpu.strides(),
             slim_cpu.dtype())
      .clone(); // Crucial: clone to own the data
}

// Overload for pointer version (used by flash attention tests)
inline at::Tensor slim_to_cpu_aten(torch::standalone::SlimTensor* slim_tensor) {
  if (!slim_tensor || slim_tensor->numel() == 0) {
    // Return an empty tensor if the slim tensor is null or has no elements.
    return at::empty(
        {0},
        at::TensorOptions().dtype(
            slim_tensor ? slim_tensor->dtype() : at::kFloat));
  }
  torch::standalone::SlimTensor slim_cpu = slim_tensor->to(at::kCPU);
  return at::from_blob(
             slim_cpu.data_ptr(),
             slim_cpu.sizes(),
             slim_cpu.strides(),
             slim_cpu.dtype())
      .clone(); // Crucial: clone to own the data
}

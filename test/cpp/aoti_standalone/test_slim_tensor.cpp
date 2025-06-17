#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <c10/core/MemoryFormat.h>
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

namespace torch::standalone {

using ::testing::ElementsAreArray;

TEST(SlimTensorInternalTest, SetSizesContiguous) {
  SlimTensor tensor = create_empty_tensor({}, {}, c10::kFloat);
  EXPECT_EQ(tensor.numel(), 1);

  std::vector<int64_t> new_size_vec = {2, 3, 4};
  c10::IntArrayRef new_sizes(new_size_vec.data(), new_size_vec.size());
  tensor.set_sizes_contiguous(new_sizes);

  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.numel(), 24);

  EXPECT_THAT(tensor.sizes(), ElementsAreArray({2, 3, 4}));
  EXPECT_THAT(tensor.strides(), ElementsAreArray({12, 4, 1}));
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST(SlimTensorInternalTest, SetSizesAndStridesNonContiguous) {
  SlimTensor tensor = create_empty_tensor({}, {}, c10::kFloat);

  // Set sizes and strides to represent a transposed tensor.
  // This is equivalent to a (2, 4) tensor that has been transposed to (4, 2).
  std::vector<int64_t> new_size_vec = {4, 2};

  // Strides of original (2, 4) were {4, 1}
  std::vector<int64_t> new_stride_vec = {1, 4};
  c10::IntArrayRef new_sizes(new_size_vec.data(), new_size_vec.size());
  c10::IntArrayRef new_strides(new_stride_vec.data(), new_stride_vec.size());

  tensor.set_sizes_and_strides(new_sizes, new_strides);

  // verify
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.numel(), 8);
  EXPECT_THAT(tensor.sizes(), ElementsAreArray({4, 2}));
  EXPECT_THAT(tensor.strides(), ElementsAreArray({1, 4}));

  // A transposed tensor is not contiguous.
  EXPECT_FALSE(tensor.is_contiguous());
}

TEST(SlimTensorInternalTest, EmptyTensorRestride) {
  SlimTensor tensor = create_empty_tensor({}, {}, c10::kFloat);
  std::vector<int64_t> size_vec = {4, 2};
  std::vector<int64_t> stride_vec = {1, 4}; // Non-contiguous strides
  tensor.set_sizes_and_strides(
      c10::IntArrayRef(size_vec.data(), size_vec.size()),
      c10::IntArrayRef(stride_vec.data(), stride_vec.size()));
  // it shouldn't be contiguous first
  EXPECT_FALSE(tensor.is_contiguous());

  // make the tensor contiguous.
  tensor.empty_tensor_restride(c10::MemoryFormat::Contiguous);

  // Sizes should NOT have changed.
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_THAT(tensor.sizes(), ElementsAreArray({4, 2}));

  // Strides SHOULD have changed to be contiguous for a (4, 2) tensor.
  EXPECT_THAT(tensor.strides(), ElementsAreArray({2, 1}));

  // The contiguity flag should now be true.
  EXPECT_TRUE(tensor.is_contiguous());
}

} // namespace torch::standalone

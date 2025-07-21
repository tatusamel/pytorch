
set(AOTI_STANDALONE_TEST_ROOT ${TORCH_ROOT}/test/cpp/aoti_standalone)

file(GLOB_RECURSE AOTI_STANDALONE_TEST_SRCS "${AOTI_STANDALONE_TEST_ROOT}/test*.cpp")

# Add FlashAttention CUDA kernel files
file(GLOB FLASH_ATTN_CUDA_SRCS
  "${TORCH_ROOT}/third_party/flash-attention/csrc/flash_attn/src/flash_fwd_hdim*_sm80.cu"
  "${TORCH_ROOT}/third_party/flash-attention/csrc/flash_attn/src/flash_fwd_split_hdim*_sm80.cu"
)

add_executable(test_aoti_standalone
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${AOTI_STANDALONE_TEST_SRCS}
  ${TORCH_ROOT}/torch/csrc/inductor/aoti_standalone/cuda/flash_attn/flash_api.cpp
  ${FLASH_ATTN_CUDA_SRCS}
)

# Add FlashAttention include directories
target_include_directories(test_aoti_standalone PRIVATE
  ${TORCH_ROOT}/third_party/flash-attention/csrc/flash_attn/src
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_aoti_standalone PRIVATE USE_GTEST)

target_link_libraries(test_aoti_standalone PRIVATE
  torch
  gtest_main
)

if(USE_CUDA)
  target_include_directories(test_aoti_standalone PRIVATE ${ATen_CUDA_INCLUDE})
  target_compile_definitions(test_aoti_standalone PRIVATE USE_CUDA)
elseif(USE_ROCM)
    target_include_directories(test_aoti_standalone PRIVATE ${ATen_HIP_INCLUDE})
    target_compile_definitions(test_aoti_standalone PRIVATE USE_ROCM)
endif()
target_compile_definitions(test_aoti_standalone PRIVATE
    CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
)

if(INSTALL_TEST)
  install(TARGETS test_aoti_standalone DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_aoti_standalone> DESTINATION bin OPTIONAL)
  endif()
endif()

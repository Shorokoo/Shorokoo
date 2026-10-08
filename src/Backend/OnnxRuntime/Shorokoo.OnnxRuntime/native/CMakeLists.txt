# The native library Shorokoo.OnnxRuntime hands ONNX Runtime: the allocator every session allocates
# through (shorokoo_ort_allocator.cpp) and the operators a CPU session registers
# (shorokoo_ort_ops.cpp). Built by Shorokoo.OnnxRuntime.csproj as part of an ordinary
# `dotnet build`; it can be built on its own too, naming the folder that holds ONNX Runtime's C API
# header (build/native/include in the Microsoft.ML.OnnxRuntime package):
#
#   cmake -S . -B build -DSHOROKOO_OUTPUT_DIR=out -DSHOROKOO_ORT_INCLUDE_DIR=<folder>
#   cmake --build build --config Release
cmake_minimum_required(VERSION 3.16)
project(shorokoo_ort_allocator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT SHOROKOO_ORT_INCLUDE_DIR OR NOT EXISTS "${SHOROKOO_ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
  message(FATAL_ERROR "SHOROKOO_ORT_INCLUDE_DIR must name the folder holding onnxruntime_c_api.h.")
endif()

add_library(shorokoo_ort_allocator SHARED shorokoo_ort_allocator.cpp shorokoo_ort_ops.cpp)
# The C API's declarations only: the runtime that loads the library hands it the API it calls.
target_include_directories(shorokoo_ort_allocator PRIVATE "${SHOROKOO_ORT_INCLUDE_DIR}")
set_target_properties(shorokoo_ort_allocator PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

if(MSVC)
  # The C runtime linked in, so the library needs no redistributable of its own beside the one
  # ONNX Runtime needs. Exceptions as C++ code throws them (/EHsc); the one this throws crosses
  # into ONNX Runtime, which catches it by type. Each float operation rounded as written (/fp:precise,
  # which contracts nothing into a fused multiply-add), so that an operator computes what the ONNX
  # operators it stands for compute, to the bit.
  set_property(TARGET shorokoo_ort_allocator PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
  target_compile_options(shorokoo_ort_allocator PRIVATE /W4 /WX /EHsc /fp:precise)
else()
  # libstdc++ linked dynamically, as ONNX Runtime's own Linux build links it: the exception this
  # throws is then of the one std::runtime_error both see. Float operations rounded as written, as
  # on Windows; a square root that sets no errno, which leaves its result as it is and lets a loop
  # taking one be vectorized.
  target_compile_options(shorokoo_ort_allocator PRIVATE -Wall -Wextra -Werror -ffp-contract=off -fno-math-errno)
  target_link_options(shorokoo_ort_allocator PRIVATE -Wl,--no-undefined)
endif()

# One folder whatever the configuration, so the build that runs this finds the library where it
# said to put it rather than under a per-configuration subfolder.
if(SHOROKOO_OUTPUT_DIR)
  set_target_properties(shorokoo_ort_allocator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "$<1:${SHOROKOO_OUTPUT_DIR}>"
    LIBRARY_OUTPUT_DIRECTORY "$<1:${SHOROKOO_OUTPUT_DIR}>")
endif()

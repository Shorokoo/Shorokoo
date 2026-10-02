# The native allocator Shorokoo.OnnxRuntime hands ONNX Runtime (see shorokoo_ort_allocator.cpp).
# Built by Shorokoo.OnnxRuntime.csproj as part of an ordinary `dotnet build`; it can be built on its
# own too:
#
#   cmake -S . -B build -DSHOROKOO_OUTPUT_DIR=out
#   cmake --build build --config Release
cmake_minimum_required(VERSION 3.16)
project(shorokoo_ort_allocator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(shorokoo_ort_allocator SHARED shorokoo_ort_allocator.cpp)
set_target_properties(shorokoo_ort_allocator PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

if(MSVC)
  # The C runtime linked in, so the library needs no redistributable of its own beside the one
  # ONNX Runtime needs. Exceptions as C++ code throws them (/EHsc); the one this throws crosses
  # into ONNX Runtime, which catches it by type.
  set_property(TARGET shorokoo_ort_allocator PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
  target_compile_options(shorokoo_ort_allocator PRIVATE /W4 /WX /EHsc)
else()
  # libstdc++ linked dynamically, as ONNX Runtime's own Linux build links it: the exception this
  # throws is then of the one std::runtime_error both see.
  target_compile_options(shorokoo_ort_allocator PRIVATE -Wall -Wextra -Werror)
  target_link_options(shorokoo_ort_allocator PRIVATE -Wl,--no-undefined)
endif()

# One folder whatever the configuration, so the build that runs this finds the library where it
# said to put it rather than under a per-configuration subfolder.
if(SHOROKOO_OUTPUT_DIR)
  set_target_properties(shorokoo_ort_allocator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "$<1:${SHOROKOO_OUTPUT_DIR}>"
    LIBRARY_OUTPUT_DIRECTORY "$<1:${SHOROKOO_OUTPUT_DIR}>")
endif()

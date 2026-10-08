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

# The same operators for ONNX Runtime's CUDA execution provider: shorokoo_ort_ops.cpp built again
# with SHOROKOO_CUDA, beside the kernel in shorokoo_adam_update.cu, into a library of its own that
# only a CUDA session registers, so that the library above needs nothing of CUDA's.
#
# SHOROKOO_CUDA_OPS: AUTO (the default) builds it where a CUDA toolkit of the major version
# ONNX Runtime's GPU package is built against (SHOROKOO_CUDA_MAJOR) is found and its nvcc compiles
# with one of the host compilers at hand, and otherwise leaves it out and says so; ON requires it;
# OFF never builds it. A CUDA session that finds no such library runs the chain of operators.
#
# nvcc compiles the kernel as a custom command rather than as CMake's CUDA language, which the
# Visual Studio generator supports only where CUDA's Visual Studio integration is installed. Its
# host compiler is SHOROKOO_CUDA_HOST_COMPILER (the folder holding cl.exe, or the path of g++)
# where given; otherwise this build's own C++ compiler and, on Windows, where nvcc refuses or
# cannot parse a Visual Studio newer than it knows, each other MSVC toolset vswhere finds.
set(SHOROKOO_CUDA_OPS AUTO CACHE STRING "Build the CUDA operators library: AUTO, ON or OFF")
set(SHOROKOO_CUDA_MAJOR 13 CACHE STRING "The CUDA major version ONNX Runtime's GPU package is built against")
set(SHOROKOO_CUDA_HOST_COMPILER "" CACHE STRING "The host compiler nvcc uses (-ccbin)")
set(SHOROKOO_CUDA_ARCHITECTURES "75;80;86;89;90;100;120" CACHE STRING
  "Compute capabilities to build machine code for; the last also as PTX for newer cards")

function(shorokoo_cuda_unavailable reason)
  if(SHOROKOO_CUDA_OPS STREQUAL "ON")
    message(FATAL_ERROR "SHOROKOO_CUDA_OPS is ON, but ${reason}")
  endif()
  message(STATUS "The CUDA operators library is not built: ${reason}")
endfunction()

if(NOT SHOROKOO_CUDA_OPS STREQUAL "OFF")
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    shorokoo_cuda_unavailable("no CUDA toolkit was found (CUDAToolkit_ROOT names one).")
  elseif(NOT CUDAToolkit_VERSION_MAJOR STREQUAL SHOROKOO_CUDA_MAJOR)
    shorokoo_cuda_unavailable("the CUDA toolkit found is ${CUDAToolkit_VERSION}, and ONNX Runtime's GPU package is built against CUDA ${SHOROKOO_CUDA_MAJOR} (CUDAToolkit_ROOT names another).")
  else()
    set(candidates "")
    if(SHOROKOO_CUDA_HOST_COMPILER)
      list(APPEND candidates "${SHOROKOO_CUDA_HOST_COMPILER}")
    elseif(MSVC)
      get_filename_component(own "${CMAKE_CXX_COMPILER}" DIRECTORY)
      list(APPEND candidates "${own}")
      set(vswhere "$ENV{ProgramFiles\(x86\)}/Microsoft Visual Studio/Installer/vswhere.exe")
      if(EXISTS "${vswhere}")
        execute_process(COMMAND "${vswhere}" -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64
                          -property installationPath
                        OUTPUT_VARIABLE installations OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REPLACE "\n" ";" installations "${installations}")
        foreach(installation IN LISTS installations)
          string(STRIP "${installation}" installation)
          file(GLOB toolsets LIST_DIRECTORIES true "${installation}/VC/Tools/MSVC/*")
          list(SORT toolsets COMPARE NATURAL ORDER DESCENDING)
          foreach(toolset IN LISTS toolsets)
            if(EXISTS "${toolset}/bin/Hostx64/x64/cl.exe")
              file(TO_CMAKE_PATH "${toolset}/bin/Hostx64/x64" folder)
              list(APPEND candidates "${folder}")
            endif()
          endforeach()
        endforeach()
      endif()
      list(REMOVE_DUPLICATES candidates)
    else()
      list(APPEND candidates "${CMAKE_CXX_COMPILER}")
    endif()

    set(probe "${CMAKE_CURRENT_BINARY_DIR}/shorokoo_cuda_probe.cu")
    file(WRITE "${probe}" "__global__ void probe(float* x) { x[threadIdx.x] += 1.0f; }\n")
    set(SHOROKOO_CUDA_CCBIN "")
    foreach(candidate IN LISTS candidates)
      execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" -c "${probe}" -o "${probe}.obj" -ccbin "${candidate}"
                      RESULT_VARIABLE probed OUTPUT_QUIET ERROR_QUIET)
      if(probed EQUAL 0)
        set(SHOROKOO_CUDA_CCBIN "${candidate}")
        break()
      endif()
    endforeach()

    if(NOT SHOROKOO_CUDA_CCBIN)
      shorokoo_cuda_unavailable("nvcc ${CUDAToolkit_VERSION} compiles with none of the host compilers ${candidates} (SHOROKOO_CUDA_HOST_COMPILER names one it supports).")
    else()
      message(STATUS "The CUDA operators library is built with nvcc ${CUDAToolkit_VERSION}, host compiler ${SHOROKOO_CUDA_CCBIN}")
      set(gencode "")
      list(GET SHOROKOO_CUDA_ARCHITECTURES -1 newest)
      foreach(arch IN LISTS SHOROKOO_CUDA_ARCHITECTURES)
        list(APPEND gencode "-gencode=arch=compute_${arch},code=sm_${arch}")
      endforeach()
      list(APPEND gencode "-gencode=arch=compute_${newest},code=compute_${newest}")
      if(MSVC)
        set(object "${CMAKE_CURRENT_BINARY_DIR}/shorokoo_adam_update.obj")
        # The static C runtime, as the libraries are linked (see above).
        set(host "-Xcompiler=/MT,/EHsc")
      else()
        set(object "${CMAKE_CURRENT_BINARY_DIR}/shorokoo_adam_update.o")
        set(host "-Xcompiler=-fPIC,-fvisibility=hidden")
      endif()
      # --fmad=false: no multiply and add contracted into one, so that each operation rounds as the
      # ONNX operator it stands for does (the kernel's intrinsics say so too).
      add_custom_command(
        OUTPUT "${object}"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" -c "${CMAKE_CURRENT_SOURCE_DIR}/shorokoo_adam_update.cu" -o "${object}"
                -ccbin "${SHOROKOO_CUDA_CCBIN}" -O3 --fmad=false -std=c++17 ${host} ${gencode}
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shorokoo_adam_update.cu" "${CMAKE_CURRENT_SOURCE_DIR}/shorokoo_adam_update.h"
        COMMENT "nvcc shorokoo_adam_update.cu"
        VERBATIM)
      set_source_files_properties("${object}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

      add_library(shorokoo_ort_cuda_ops SHARED shorokoo_ort_ops.cpp "${object}")
      target_compile_definitions(shorokoo_ort_cuda_ops PRIVATE SHOROKOO_CUDA)
      target_include_directories(shorokoo_ort_cuda_ops PRIVATE "${SHOROKOO_ORT_INCLUDE_DIR}")
      # The CUDA runtime linked in: the kernel launches on the stream ONNX Runtime hands it, in the
      # device's primary context, which every CUDA runtime in the process shares.
      target_link_libraries(shorokoo_ort_cuda_ops PRIVATE CUDA::cudart_static)
      set_target_properties(shorokoo_ort_cuda_ops PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
      if(MSVC)
        set_property(TARGET shorokoo_ort_cuda_ops PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
        target_compile_options(shorokoo_ort_cuda_ops PRIVATE /W4 /WX /EHsc /fp:precise)
      else()
        target_compile_options(shorokoo_ort_cuda_ops PRIVATE -Wall -Wextra -Werror -ffp-contract=off -fno-math-errno)
        target_link_options(shorokoo_ort_cuda_ops PRIVATE -Wl,--no-undefined)
      endif()
      if(SHOROKOO_OUTPUT_DIR)
        set_target_properties(shorokoo_ort_cuda_ops PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY "$<1:${SHOROKOO_OUTPUT_DIR}>"
          LIBRARY_OUTPUT_DIRECTORY "$<1:${SHOROKOO_OUTPUT_DIR}>")
      endif()
    endif()
  endif()
endif()

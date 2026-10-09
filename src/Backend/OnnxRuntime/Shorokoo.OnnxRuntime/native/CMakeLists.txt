# The native library Shorokoo.OnnxRuntime hands ONNX Runtime: the allocator every session allocates
# through (shorokoo_ort_allocator.cpp) and the operators a CPU session registers
# (shorokoo_ort_ops.cpp). Built by Shorokoo.OnnxRuntime.csproj as part of an ordinary
# `dotnet build`; it can be built on its own too, naming the folder that holds ONNX Runtime's C API
# header (build/native/include in the Microsoft.ML.OnnxRuntime package):
#
#   cmake -S . -B build -DSHOROKOO_OUTPUT_DIR=out -DSHOROKOO_ORT_INCLUDE_DIR=<folder>
#   cmake --build build --config Release
#
# The same operators' CUDA build, a library of its own, is made beside it where a CUDA toolkit is
# at hand (see SHOROKOO_CUDA_OPS below).
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
# only a CUDA session registers, so that the library above needs nothing of CUDA's. The CUDA runtime
# is linked in statically, so the library needs nothing beside the operating system's own libraries
# and the card's driver.
#
# SHOROKOO_CUDA_OPS: AUTO (the default) builds it where a CUDA toolkit of the major version
# ONNX Runtime's GPU package is built against (SHOROKOO_CUDA_MAJOR) is found and its nvcc compiles
# the kernel with one of the host compilers at hand, and otherwise leaves it out with a status
# message; ON requires it, and stops the build saying what was missing; OFF never builds it. A CUDA
# session that finds no such library runs the chain of operators the update is written as.
# CUDAToolkit_ROOT (a variable or an environment variable) names the toolkit where nvcc is not on
# the PATH.
#
# nvcc compiles the kernel as a custom command rather than as CMake's CUDA language, which the
# Visual Studio generator supports only where CUDA's Visual Studio integration is installed. Its
# host compiler is SHOROKOO_CUDA_HOST_COMPILER (the folder holding cl.exe, or the path of g++) where
# given. Otherwise it is the first that compiles the kernel of: this build's own C++ compiler, then,
# on Windows, each other MSVC toolset vswhere finds, newest first, and on Linux each g++-<version>
# in /usr/bin, newest first. Each nvcc release accepts only the host compilers it knows (CUDA 13.0
# and 13.1 the MSVC of Visual Studio 2019 and 2022, CUDA 13.2 and later that of Visual Studio 2026
# too), so a machine whose newest Visual Studio is newer than its nvcc still builds the library with
# an older toolset beside it.
set(SHOROKOO_CUDA_OPS AUTO CACHE STRING "Build the CUDA operators library: AUTO, ON or OFF")
set_property(CACHE SHOROKOO_CUDA_OPS PROPERTY STRINGS AUTO ON OFF)
set(SHOROKOO_CUDA_MAJOR 13 CACHE STRING "The CUDA major version ONNX Runtime's GPU package is built against")
set(SHOROKOO_CUDA_HOST_COMPILER "" CACHE STRING "The host compiler nvcc uses (-ccbin)")
set(SHOROKOO_CUDA_ARCHITECTURES "75;80;86;89;90;100;120" CACHE STRING
  "Compute capabilities to build machine code for; the first also as PTX, which every newer card compiles")

string(TOUPPER "${SHOROKOO_CUDA_OPS}" shorokoo_cuda_ops)
if(NOT shorokoo_cuda_ops MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "SHOROKOO_CUDA_OPS is '${SHOROKOO_CUDA_OPS}'; it takes AUTO, ON or OFF.")
endif()

if(WIN32)
  set(shorokoo_cuda_ops_file "shorokoo_ort_cuda_ops.dll")
else()
  set(shorokoo_cuda_ops_file "libshorokoo_ort_cuda_ops.so")
endif()

# Leaves the library out: an error where it is required, a status message otherwise. A copy an
# earlier configuration built is removed from the output, which would otherwise be deployed as if
# this one had built it.
function(shorokoo_cuda_unavailable reason)
  if(shorokoo_cuda_ops STREQUAL "ON")
    message(FATAL_ERROR "SHOROKOO_CUDA_OPS is ON, but ${reason}")
  endif()
  message(STATUS "The CUDA operators library is not built: ${reason}")
  if(SHOROKOO_OUTPUT_DIR)
    file(REMOVE "${SHOROKOO_OUTPUT_DIR}/${shorokoo_cuda_ops_file}")
  endif()
endfunction()

if(shorokoo_cuda_ops STREQUAL "OFF")
  shorokoo_cuda_unavailable("SHOROKOO_CUDA_OPS is OFF.")
else()
  # The toolkit is looked for afresh unless the folders naming it are those the cache was filled
  # from; what was found before would otherwise stay in the cache.
  set(root_key "${CUDAToolkit_ROOT}|$ENV{CUDAToolkit_ROOT}|$ENV{CUDA_PATH}")
  if(NOT DEFINED SHOROKOO_CUDA_ROOT_KEY OR NOT SHOROKOO_CUDA_ROOT_KEY STREQUAL root_key)
    get_cmake_property(cached CACHE_VARIABLES)
    foreach(variable IN LISTS cached)
      if(variable MATCHES "^(_cmake_CUDAToolkit_|CUDAToolkit_|CUDA_)" AND NOT variable STREQUAL "CUDAToolkit_ROOT")
        unset(${variable} CACHE)
      endif()
    endforeach()
  endif()
  set(SHOROKOO_CUDA_ROOT_KEY "${root_key}" CACHE INTERNAL "The CUDA toolkit folders the last configuration was named")
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND OR NOT CUDAToolkit_NVCC_EXECUTABLE OR NOT TARGET CUDA::cudart_static)
    shorokoo_cuda_unavailable("no CUDA toolkit with nvcc and the static CUDA runtime was found (CUDAToolkit_ROOT names one).")
  elseif(NOT CUDAToolkit_VERSION_MAJOR STREQUAL SHOROKOO_CUDA_MAJOR)
    shorokoo_cuda_unavailable("the CUDA toolkit found is ${CUDAToolkit_VERSION}, and ONNX Runtime's GPU package is built against CUDA ${SHOROKOO_CUDA_MAJOR} (CUDAToolkit_ROOT names another toolkit).")
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
        set(toolsets "")
        foreach(installation IN LISTS installations)
          string(STRIP "${installation}" installation)
          file(GLOB found LIST_DIRECTORIES true "${installation}/VC/Tools/MSVC/*")
          list(APPEND toolsets ${found})
        endforeach()
        # Newest first across every installation: the folder names are the toolset versions.
        set(keyed "")
        foreach(toolset IN LISTS toolsets)
          get_filename_component(version "${toolset}" NAME)
          list(APPEND keyed "${version}|${toolset}")
        endforeach()
        list(SORT keyed COMPARE NATURAL ORDER DESCENDING)
        foreach(entry IN LISTS keyed)
          string(REGEX REPLACE "^[^|]*\\|" "" toolset "${entry}")
          if(EXISTS "${toolset}/bin/Hostx64/x64/cl.exe")
            file(TO_CMAKE_PATH "${toolset}/bin/Hostx64/x64" folder)
            list(APPEND candidates "${folder}")
          endif()
        endforeach()
      endif()
    else()
      list(APPEND candidates "${CMAKE_CXX_COMPILER}")
      file(GLOB versioned "/usr/bin/g++-[0-9]*")
      list(SORT versioned COMPARE NATURAL ORDER DESCENDING)
      list(APPEND candidates ${versioned})
    endif()
    list(REMOVE_DUPLICATES candidates)

    if(MSVC)
      set(object_suffix ".obj")
      # The static C runtime, as the libraries are linked (see above). nvcc sets up the environment
      # of the toolset it is handed itself, so none of another toolset's may reach it.
      set(host_flags "-Xcompiler=/MT,/EHsc")
      set(nvcc_launch "${CMAKE_COMMAND}" -E env --unset=INCLUDE --unset=LIB --unset=LIBPATH --unset=EXTERNAL_INCLUDE)
    else()
      set(object_suffix ".o")
      set(host_flags "-Xcompiler=-fPIC,-fvisibility=hidden")
      set(nvcc_launch "")
    endif()
    # --fmad=false: no multiply and add contracted into one, so that each operation rounds as the
    # ONNX operator it stands for does (the kernel's intrinsics say so too).
    set(nvcc_flags -O3 --fmad=false -std=c++17 ${host_flags})

    # The kernel itself, compiled for one architecture with each candidate until one succeeds: a
    # host compiler nvcc refuses, or whose headers it cannot parse, fails here rather than in the
    # build.
    list(GET SHOROKOO_CUDA_ARCHITECTURES 0 oldest)
    set(probe "${CMAKE_CURRENT_BINARY_DIR}/shorokoo_cuda_probe${object_suffix}")
    set(SHOROKOO_CUDA_CCBIN "")
    set(refusals "")
    foreach(candidate IN LISTS candidates)
      execute_process(COMMAND ${nvcc_launch} "${CUDAToolkit_NVCC_EXECUTABLE}" -c "${CMAKE_CURRENT_SOURCE_DIR}/shorokoo_adam_update.cu"
                        -o "${probe}" -ccbin "${candidate}" ${nvcc_flags} "-gencode=arch=compute_${oldest},code=sm_${oldest}"
                      RESULT_VARIABLE probed OUTPUT_VARIABLE said ERROR_VARIABLE said)
      if(probed EQUAL 0)
        set(SHOROKOO_CUDA_CCBIN "${candidate}")
        break()
      endif()
      string(REGEX MATCH "[^\n]*(error|Error|ERROR)[^\n]*" why "${said}")
      if(NOT why)
        string(STRIP "${said}" why)
      endif()
      string(APPEND refusals "\n  ${candidate}: ${why}")
    endforeach()
    file(REMOVE "${probe}")

    if(NOT SHOROKOO_CUDA_CCBIN)
      shorokoo_cuda_unavailable("nvcc ${CUDAToolkit_VERSION} (${CUDAToolkit_NVCC_EXECUTABLE}) compiles the kernel with none of the host compilers at hand:${refusals}\nEach CUDA release supports the compilers and system headers its installation guide lists, and a later CUDA ${SHOROKOO_CUDA_MAJOR} release newer ones: install a compiler this nvcc supports and name it with SHOROKOO_CUDA_HOST_COMPILER, or name a later toolkit with CUDAToolkit_ROOT.")
    else()
      message(STATUS "The CUDA operators library is built with nvcc ${CUDAToolkit_VERSION}, host compiler ${SHOROKOO_CUDA_CCBIN}")
      set(gencode "")
      foreach(arch IN LISTS SHOROKOO_CUDA_ARCHITECTURES)
        list(APPEND gencode "-gencode=arch=compute_${arch},code=sm_${arch}")
      endforeach()
      list(APPEND gencode "-gencode=arch=compute_${oldest},code=compute_${oldest}")
      set(object "${CMAKE_CURRENT_BINARY_DIR}/shorokoo_adam_update${object_suffix}")
      add_custom_command(
        OUTPUT "${object}"
        COMMAND ${nvcc_launch} "${CUDAToolkit_NVCC_EXECUTABLE}" -c "${CMAKE_CURRENT_SOURCE_DIR}/shorokoo_adam_update.cu" -o "${object}"
                -ccbin "${SHOROKOO_CUDA_CCBIN}" ${nvcc_flags} ${gencode}
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
        # The static runtime's symbols kept inside the library, so that none of them stands in for
        # the shared CUDA runtime ONNX Runtime's CUDA provider loads.
        target_link_options(shorokoo_ort_cuda_ops PRIVATE -Wl,--no-undefined -Wl,--exclude-libs,ALL)
      endif()
      if(SHOROKOO_OUTPUT_DIR)
        set_target_properties(shorokoo_ort_cuda_ops PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY "$<1:${SHOROKOO_OUTPUT_DIR}>"
          LIBRARY_OUTPUT_DIRECTORY "$<1:${SHOROKOO_OUTPUT_DIR}>")
      endif()
    endif()
  endif()
endif()

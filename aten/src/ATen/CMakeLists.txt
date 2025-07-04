cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

IF(NOT MSVC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-qualifiers")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-ignored-qualifiers")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-absolute-value")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-absolute-value")
ENDIF(NOT MSVC)

# Can be compiled standalone
IF(NOT AT_INSTALL_BIN_DIR OR NOT AT_INSTALL_LIB_DIR OR NOT AT_INSTALL_INCLUDE_DIR OR NOT AT_INSTALL_SHARE_DIR)
  SET(AT_INSTALL_BIN_DIR "bin" CACHE PATH "AT install binary subdirectory")
  SET(AT_INSTALL_LIB_DIR "lib" CACHE PATH "AT install library subdirectory")
  SET(AT_INSTALL_INCLUDE_DIR "include" CACHE PATH "AT install include subdirectory")
  SET(AT_INSTALL_SHARE_DIR "share" CACHE PATH "AT install include subdirectory")
ENDIF()

CONFIGURE_FILE(Config.h.in "${CMAKE_CURRENT_SOURCE_DIR}/Config.h")
CONFIGURE_FILE(cuda/CUDAConfig.h.in "${CMAKE_CURRENT_SOURCE_DIR}/cuda/CUDAConfig.h")

# 注意：如果修改了这些文件匹配规则(globs)，
#       必须同步更新 setup.py 中的 package_data 配置
# -----------------------------------------------------------
# 基础 CPU 相关文件 (所有平台通用)
# -----------------------------------------------------------
# 收集基础头文件:
#   - 当前目录所有 .h 文件
#   - detail/ 子目录所有 .h 文件
#   - cpu/ 子目录所有 .h 文件
FILE(GLOB base_h "*.h" "detail/*.h" "cpu/*.h")

# 收集基础 C++ 实现文件
FILE(GLOB base_cpp "*.cpp" "detail/*.cpp" "cpu/*.cpp")

# 添加核心代码子目录
add_subdirectory(core)

# -----------------------------------------------------------
# CUDA 相关文件 (NVIDIA GPU支持)
# -----------------------------------------------------------
# CUDA 头文件:
#   - cuda/ 目录的头文件和 .cuh (CUDA头文件)
#   - 包含子目录 detail/
FILE(GLOB cuda_h "cuda/*.h" "cuda/detail/*.h" "cuda/*.cuh" "cuda/detail/*.cuh")

# CUDA C++ 实现文件 (.cpp)
FILE(GLOB cuda_cpp "cuda/*.cpp" "cuda/detail/*.cpp")

# CUDA 内核文件 (.cu)
FILE(GLOB cuda_cu "cuda/*.cu" "cuda/detail/*.cu")

# -----------------------------------------------------------
# cuDNN 相关文件 (NVIDIA 深度神经网络库)
# -----------------------------------------------------------
FILE(GLOB cudnn_h "cudnn/*.h" "cudnn/*.cuh")    # cuDNN 头文件
FILE(GLOB cudnn_cpp "cudnn/*.cpp")               # cuDNN C++ 实现

# -----------------------------------------------------------
# MIOpen 相关文件 (AMD ROCm 的深度学习库)
# -----------------------------------------------------------
FILE(GLOB miopen_h "miopen/*.h")                 # MIOpen 头文件
FILE(GLOB miopen_cpp "miopen/*.cpp")             # MIOpen C++ 实现

# -----------------------------------------------------------
# CPU 加速库
# -----------------------------------------------------------
# Intel MKL (数学核心库)
FILE(GLOB mkl_cpp "mkl/*.cpp")

# Intel oneDNN (原MKLDNN，深度神经网络库)
FILE(GLOB mkldnn_cpp "mkldnn/*.cpp")

# -----------------------------------------------------------
# 原生操作实现 (按后端分类)
# -----------------------------------------------------------
# 通用原生操作
FILE(GLOB native_cpp "native/*.cpp")

# 稀疏张量操作
FILE(GLOB native_sparse_cpp "native/sparse/*.cpp")              # CPU 稀疏操作
FILE(GLOB native_sparse_cuda_cu "native/sparse/cuda/*.cu")      # CUDA 稀疏内核
FILE(GLOB native_sparse_cuda_cpp "native/sparse/cuda/*.cpp")    # CUDA 稀疏包装

# 各加速库的原生实现
FILE(GLOB native_cudnn_cpp "native/cudnn/*.cpp")    # cuDNN 封装实现
FILE(GLOB native_miopen_cpp "native/miopen/*.cpp")  # MIOpen 封装实现
FILE(GLOB native_cuda_cu "native/cuda/*.cu")        # CUDA 内核实现
FILE(GLOB native_cuda_cpp "native/cuda/*.cpp")      # CUDA 包装实现
FILE(GLOB native_mkl_cpp "native/mkl/*.cpp")        # MKL 封装实现
FILE(GLOB native_mkldnn_cpp "native/mkldnn/*.cpp")  # oneDNN 封装实现

set(all_cpu_cpp ${base_cpp} ${ATen_CORE_SRCS} ${native_cpp} ${native_sparse_cpp} ${native_mkl_cpp} ${native_mkldnn_cpp} ${generated_cpp} ${ATen_CPU_SRCS} ${cpu_kernel_cpp})
if(AT_MKL_ENABLED)
  set(all_cpu_cpp ${all_cpu_cpp} ${mkl_cpp})
endif()
if(AT_MKLDNN_ENABLED)
  set(all_cpu_cpp ${all_cpu_cpp} ${mkldnn_cpp})
endif()

IF(USE_CUDA OR USE_ROCM)
  list(APPEND ATen_CUDA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
  set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS} ${cuda_cu} ${native_cuda_cu} ${native_sparse_cuda_cu})
  set(all_cuda_cpp ${native_sparse_cuda_cpp} ${cuda_cpp} ${native_cuda_cpp} ${cuda_generated_cpp} ${ATen_CUDA_SRCS})
  IF(USE_CUDA)
    SET(all_cuda_cpp ${native_cudnn_cpp} ${native_miopen_cpp} ${all_cuda_cpp})
    IF(CUDNN_FOUND)
      SET(all_cuda_cpp ${all_cuda_cpp} ${cudnn_cpp})
    ENDIF()
  ELSEIF(USE_ROCM)
    SET(all_cuda_cpp ${native_cudnn_cpp} ${native_miopen_cpp} ${miopen_cpp} ${all_cuda_cpp})
  ENDIF()
endif()

filter_list(generated_h generated_cpp "\\.h$")
filter_list(cuda_generated_h cuda_generated_cpp "\\.h$")

list(APPEND ATen_CPU_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/..)
# so the build can find the generated header files
list(APPEND ATen_CPU_INCLUDE ${CMAKE_CURRENT_BINARY_DIR})
IF(NOT AT_LINK_STYLE)
  SET(AT_LINK_STYLE SHARED)
ENDIF()

IF(BLAS_FOUND)
  IF ($ENV{TH_BINARY_BUILD})
    MESSAGE(STATUS "TH_BINARY_BUILD detected. Enabling special linkage.")
    list(APPEND ATen_CPU_DEPENDENCY_LIBS
      "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
    if(USE_CUDA OR USE_ROCM)
      list(APPEND ATen_CUDA_DEPENDENCY_LIBS
        "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
    endif()
  ELSE ($ENV{TH_BINARY_BUILD})
    list(APPEND ATen_CPU_DEPENDENCY_LIBS ${BLAS_LIBRARIES})
    if(USE_CUDA OR USE_ROCM)
      list(APPEND ATen_CUDA_DEPENDENCY_LIBS "${BLAS_LIBRARIES}")
    endif()
  ENDIF ($ENV{TH_BINARY_BUILD})
ENDIF(BLAS_FOUND)

IF(LAPACK_FOUND)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  if(USE_CUDA OR USE_ROCM)
    # Although Lapack provides CPU (and thus, one might expect that ATen_cuda
    # would not need this at all), some of our libraries (magma in particular)
    # backend to CPU BLAS/LAPACK implementations, and so it is very important
    # we get the *right* implementation, because even if the symbols are the
    # same, LAPACK implementions may have different calling conventions.
    # This caused https://github.com/pytorch/pytorch/issues/7353
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  endif()
ENDIF(LAPACK_FOUND)

IF (UNIX AND NOT APPLE)
   INCLUDE(CheckLibraryExists)
   # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
   CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_LIBRT)
   IF(NEED_LIBRT)
     list(APPEND ATen_CPU_DEPENDENCY_LIBS rt)
     SET(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} rt)
   ENDIF(NEED_LIBRT)
ENDIF(UNIX AND NOT APPLE)

IF(UNIX)
  SET(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
  CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
  IF(HAVE_MMAP)
    ADD_DEFINITIONS(-DHAVE_MMAP=1)
  ENDIF(HAVE_MMAP)
  # done for lseek: https://www.gnu.org/software/libc/manual/html_node/File-Position-Primitive.html
  ADD_DEFINITIONS(-D_FILE_OFFSET_BITS=64)
  CHECK_FUNCTION_EXISTS(shm_open HAVE_SHM_OPEN)
  IF(HAVE_SHM_OPEN)
    ADD_DEFINITIONS(-DHAVE_SHM_OPEN=1)
  ENDIF(HAVE_SHM_OPEN)
  CHECK_FUNCTION_EXISTS(shm_unlink HAVE_SHM_UNLINK)
  IF(HAVE_SHM_UNLINK)
    ADD_DEFINITIONS(-DHAVE_SHM_UNLINK=1)
  ENDIF(HAVE_SHM_UNLINK)
  CHECK_FUNCTION_EXISTS(malloc_usable_size HAVE_MALLOC_USABLE_SIZE)
  IF(HAVE_MALLOC_USABLE_SIZE)
    ADD_DEFINITIONS(-DHAVE_MALLOC_USABLE_SIZE=1)
  ENDIF(HAVE_MALLOC_USABLE_SIZE)
ENDIF(UNIX)

if(NOT MSVC)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS m)
endif()

if(MKLDNN_FOUND)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS ${MKLDNN_LIBRARIES})
endif(MKLDNN_FOUND)

list(APPEND ATen_CPU_DEPENDENCY_LIBS cpuinfo)

if(NOT MSVC AND NOT EMSCRIPTEN)
  # Preserve values for the main build
  set(__aten_sleef_build_shared_libs ${BUILD_SHARED_LIBS})
  set(__aten_sleef_build_tests ${BUILD_TESTS})

  # Unset our restrictive C++ flags here and reset them later.
  # Remove this once we use proper target_compile_options.
  set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS)

  # Bump up optimization level for sleef to -O1, since at -O0 the compiler
  # excessively spills intermediate vector registers to the stack
  # and makes things run impossibly slowly
  set(OLD_CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
  IF(${CMAKE_C_FLAGS_DEBUG} MATCHES "-O0")
    string(REGEX REPLACE "-O0" "-O1" CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
  ELSE()
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1")
  ENDIF()

  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build sleef static" FORCE)
  set(BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
  set(BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
  set(BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/sleef" ${CMAKE_BINARY_DIR}/sleef)
  set_property(TARGET sleef PROPERTY FOLDER "dependencies")
  list(APPEND ATen_THIRD_PARTY_INCLUDE ${CMAKE_BINARY_DIR}/include)
  link_directories(${CMAKE_BINARY_DIR}/sleef/lib)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS sleef)

  set(CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
  set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})

  # Set these back. TODO: Use SLEEF_ to pass these instead
  set(BUILD_SHARED_LIBS ${__aten_sleef_build_shared_libs} CACHE BOOL "Build shared libs" FORCE)
  set(BUILD_TESTS ${__aten_sleef_build_tests} CACHE BOOL "Build tests" FORCE)
endif()

IF(USE_CUDA AND NOT USE_ROCM)
  IF ($ENV{ATEN_STATIC_CUDA})
    # CuFFT has a complicated static story (especially around CUDA < 9) because it has device callback support
    # we first have to build a fake lib that links with no device callbacks,
    # and then we link against this object file.
    # This was recommended by the CuFFT team at NVIDIA

    # build fake CuFFT lib in build dir
    EXECUTE_PROCESS(COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/empty_file.cc)
    if(${CUDA_VERSION_MAJOR} EQUAL "8")
      SET(CUFFT_FAKELINK_OPTIONS
	--generate-code arch=compute_35,code=sm_35
	--generate-code arch=compute_50,code=sm_50
	--generate-code arch=compute_60,code=sm_60)
    elseif(${CUDA_VERSION_MAJOR} EQUAL "9")
      SET(CUFFT_FAKELINK_OPTIONS
	--generate-code arch=compute_35,code=sm_35
	--generate-code arch=compute_50,code=sm_50
	--generate-code arch=compute_60,code=sm_60
	--generate-code arch=compute_70,code=sm_70)
    elseif(${CUDA_VERSION_MAJOR} EQUAL "10")
      SET(CUFFT_FAKELINK_OPTIONS
	--generate-code arch=compute_35,code=sm_35
	--generate-code arch=compute_50,code=sm_50
	--generate-code arch=compute_60,code=sm_60
	--generate-code arch=compute_70,code=sm_70)
    else()
      MESSAGE(FATAL_ERROR "Unhandled major cuda version ${CUDA_VERSION_MAJOR}")
    endif()
    ADD_CUSTOM_COMMAND(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a
      COMMAND "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" -o ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a -Xcompiler -fPIC
      ${CUFFT_FAKELINK_OPTIONS}
      --device-link ${CMAKE_CURRENT_BINARY_DIR}/empty_file.cc -lcufft_static -lculibos
      )
    ADD_CUSTOM_TARGET(FAKELINKED_CUFFT_TARGET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a)
    add_library(FAKELINKED_CUFFT STATIC IMPORTED GLOBAL)
    add_dependencies(FAKELINKED_CUFFT FAKELINKED_CUFFT_TARGET)
    set_target_properties(FAKELINKED_CUFFT PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a)

    list(APPEND ATen_CUDA_DEPENDENCY_LIBS
      ${CUDA_LIBRARIES}
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusparse_static.a
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand_static.a
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a
      FAKELINKED_CUFFT
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft_static.a
      )
  ELSE()
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS
      ${CUDA_LIBRARIES}
      ${CUDA_cusparse_LIBRARY}
      ${CUDA_curand_LIBRARY})
  ENDIF()

  if(CUDNN_FOUND)
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${CUDNN_LIBRARIES})
  endif(CUDNN_FOUND)

  IF(USE_MAGMA)
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${MAGMA_LIBRARIES})
    IF ($ENV{TH_BINARY_BUILD})
      list(APPEND ATen_CUDA_DEPENDENCY_LIBS
        "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
    ENDIF($ENV{TH_BINARY_BUILD})
  ENDIF(USE_MAGMA)
  IF ($ENV{ATEN_STATIC_CUDA})
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a")
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart_static.a")
  ENDIF($ENV{ATEN_STATIC_CUDA})
ENDIF()

IF(USE_ROCM)
 ### Link in the ROCm libraries BLAS / RNG.
 FIND_LIBRARY(ROCBLAS_LIBRARY rocblas HINTS ${ROCBLAS_PATH}/lib)
 FIND_LIBRARY(HIPRAND_LIBRARY hiprand HINTS ${HIPRAND_PATH}/lib)

 list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${ROCBLAS_LIBRARY} ${HIPRAND_LIBRARY})
ENDIF()

# Include CPU paths for CUDA as well
list(APPEND ATen_CUDA_INCLUDE ${ATen_CPU_INCLUDE})

# We have two libraries: libATen_cpu.so and libATen_cuda.so,
# with libATen_cuda.so depending on libATen_cpu.so.  The CPU library
# contains CPU code only.  libATen_cpu.so is invariant to the setting
# of USE_CUDA (it always builds the same way); libATen_cuda.so is only
# built when USE_CUDA=1 and CUDA is available.
set(ATen_CPU_SRCS ${all_cpu_cpp})
if(AT_LINK_STYLE STREQUAL "INTERFACE")
  # Source code can't be added to an interface library, so it is
  # passed back to be compiled into the containing library
  add_library(ATen_cpu INTERFACE)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS ATEN_CPU_FILES_GEN_LIB)
else()
  add_library(ATen_cpu ${AT_LINK_STYLE} ${ATen_CPU_SRCS})
  if (ATen_THIRD_PARTY_INCLUDE)
    target_include_directories(ATen_cpu SYSTEM PRIVATE ${ATen_THIRD_PARTY_INCLUDE})
  endif()
  target_include_directories(ATen_cpu INTERFACE $<INSTALL_INTERFACE:include>)
  target_include_directories(ATen_cpu PRIVATE ${ATen_CPU_INCLUDE})
  target_link_libraries(ATen_cpu PUBLIC ${ATen_CPU_DEPENDENCY_LIBS})
  target_link_libraries(ATen_cpu PRIVATE ATEN_CPU_FILES_GEN_LIB)
  caffe2_interface_library(ATen_cpu ATen_cpu_library)
  # Set standard properties on the target
  torch_set_target_props(ATen_cpu)

  # Make sure these don't get built by parent
  set(ATen_CPU_SRCS)
endif()

if(USE_CUDA OR USE_ROCM)
  set(ATen_CUDA_SRCS ${all_cuda_cpp})
  if(AT_LINK_STYLE STREQUAL "INTERFACE")
    # Source code can't be added to an interface library, so it is
    # passed back to be compiled into the containing library
    add_library(ATen_cuda INTERFACE)
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ATEN_CUDA_FILES_GEN_LIB)
  else()
    # A hack to deal with cuda library dependencies and modern CMake: the
    # CUDA_ADD_LIBRARY includes a target_link_libraries, and as a result,
    # one cannot use PUBLIC/PRIVATE/INTERFACE for the target anymore. This
    # hack adds the PRIVATE keywords to CUDA_LIBRARIES so we can deal with
    # it. We will then manually add the cudart library as interface libs.
    set(__tmp ${CUDA_LIBRARIES})
    set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})
    torch_cuda_based_add_library(ATen_cuda ${AT_LINK_STYLE} ${ATen_CUDA_SRCS})
    set(CUDA_LIBRARIES ${__tmp})
    target_link_libraries(ATen_cuda INTERFACE caffe2::cudart)

    target_include_directories(
        ATen_cuda INTERFACE $<INSTALL_INTERFACE:include>)
    target_include_directories(
        ATen_cuda PRIVATE ${ATen_THIRD_PARTY_INCLUDE})
    target_include_directories(
        ATen_cuda PRIVATE ${ATen_CUDA_INCLUDE})
    target_link_libraries(
        ATen_cuda PRIVATE ${ATen_CUDA_DEPENDENCY_LIBS} ATEN_CUDA_FILES_GEN_LIB)

    # These public dependencies must go after the previous dependencies, as the
    # order of the libraries in the linker call matters here when statically
    # linking; libculibos and cublas must be last.
    target_link_libraries(
        ATen_cuda PUBLIC ATen_cpu ${ATen_PUBLIC_CUDA_DEPENDENCY_LIBS})

    # Set standard properties on the target
    torch_set_target_props(ATen_cuda)

    caffe2_interface_library(ATen_cuda ATen_cuda_library)

    # Make sure these don't get built by parent
    set(ATen_CUDA_SRCS)
  endif()
endif()

if(NOT AT_LINK_STYLE STREQUAL "INTERFACE")
  if(USE_CUDA)
    if (NOT $ENV{ATEN_STATIC_CUDA})
      cuda_add_cublas_to_target(ATen_cuda)
      cuda_add_cufft_to_target(ATen_cuda)
    endif()
  endif()

  if(NOT MSVC)
    torch_compile_options(ATen_cpu)
    if(USE_CUDA OR USE_ROCM)
      torch_compile_options(ATen_cuda)
    endif()
  endif()

  if(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
    set_property(TARGET ATen_cpu PROPERTY CXX_STANDARD 11)
    if(USE_CUDA OR USE_ROCM)
      set_property(TARGET ATen_cuda PROPERTY CXX_STANDARD 11)
    endif()
  endif()
endif()

SET(ATEN_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${AT_INSTALL_INCLUDE_DIR}")
CONFIGURE_FILE(ATenConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/ATenConfig.cmake")
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/ATenConfig.cmake"
  DESTINATION "${AT_INSTALL_SHARE_DIR}/cmake/ATen")

# https://stackoverflow.com/questions/11096471/how-can-i-install-a-hierarchy-of-files-using-cmake
FOREACH(HEADER ${base_h} ${ATen_CORE_HEADERS} ${cuda_h} ${cudnn_h})
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" HEADER_SUB ${HEADER})
  GET_FILENAME_COMPONENT(DIR ${HEADER_SUB} DIRECTORY)
  INSTALL(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen/${DIR})
ENDFOREACH()
FOREACH(HEADER ${generated_h} ${cuda_generated_h})
  # NB: Assumed to be flat
  INSTALL(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen)
ENDFOREACH()
INSTALL(FILES ${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml
  DESTINATION ${AT_INSTALL_SHARE_DIR}/ATen)

if(ATEN_NO_TEST)
  message("disable test because ATEN_NO_TEST is set")
else()
  add_subdirectory(test)
endif()

# Pass source, includes, and libs to parent
# 将当前作用域的变量值提升到父作用域
# 这样父作用域的CMake脚本可以使用这些变量值

# 核心源文件列表（跨平台基础代码）
set(ATen_CORE_SRCS ${ATen_CORE_SRCS} PARENT_SCOPE)

# CPU特定源文件列表（仅CPU后端的实现）
set(ATen_CPU_SRCS ${ATen_CPU_SRCS} PARENT_SCOPE)

# CUDA特定源文件列表（GPU加速的实现）
set(ATen_CUDA_SRCS ${Atn_CUDA_SRCS} PARENT_SCOPE)

# CPU测试源文件列表（CPU后端的测试代码）
set(ATen_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS} PARENT_SCOPE)

# CUDA测试源文件列表（GPU后端的测试代码）
set(ATen_CUDA_TEST_SRCS ${ATen_CUDA_TEST_SRCS} PARENT_SCOPE)

# CPU头文件包含路径（CPU后端的头文件目录）
set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE} PARENT_SCOPE)

# 第三方库包含路径（依赖的第三方库头文件目录）
set(ATen_THIRD_PARTY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE} PARENT_SCOPE)

# CUDA头文件包含路径（CUDA相关的头文件目录）
set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE} PARENT_SCOPE)

# CPU依赖库列表（CPU后端需要链接的库）
set(ATen_CPU_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS} PARENT_SCOPE)

# CUDA依赖库列表（CUDA后端需要链接的库）
set(ATen_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS} PARENT_SCOPE)

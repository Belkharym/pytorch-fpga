# Build file for C10 OpenCL
#
# C10 OpenCL is a minimal library, but it does depend on OpenCL.

include(../../cmake/public/utils.cmake)

# ---[ Configure macro file.
set(C10_OPENCL_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
set(_OPENCL_KERNEL_DIR opencl/kernels)
set(_tmp_OPENCL_KERNEL_DIR ${CMAKE_INSTALL_PREFIX}/${_OPENCL_KERNEL_DIR})
if(USE_FPGA)
  set(C10_USE_FPGA ${USE_FPGA})
endif()
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/impl/opencl_cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/opencl/impl/opencl_cmake_macros.h)

# NB: All previous cu files are renamed into cc files.  This isn't tested at the
# moment.
file(GLOB C10_OPENCL_SRCS
    *.cpp
    impl/*.cpp
    )

message(STATUS "C10_OPENCL_SRCS ${C10_OPENCL_SRCS}")

file(GLOB C10_OPENCL_HEADERS
    *.h
    impl/*.h
)

message(STATUS "C10_OPENCL_HEADERS ${C10_OPENCL_HEADERS}")

if(USE_FPGA) # Save current compiler and switch with the FPGA one.
  set(tmp_CXX_COMPILER ${CMAKE_CXX_COMPILER})
  set(CMAKE_CXX_COMPILER ${SDACCEL_XCPP})
endif()
add_library(c10_opencl ${C10_OPENCL_SRCS} ${C10_OPENCL_HEADERS})
if(USE_FPGA) # Restore original compiler
  set(CMAKE_CXX_COMPILER ${tmp_CXX_COMPILER})
  set(tmp_CXX_COMPILER)
endif()
# If building shared library, set dllimport/dllexport proper.
target_compile_options(c10_opencl PRIVATE "-DC10_OPENCL_BUILD_MAIN_LIB")
# Enable hidden visibility if compiler supports it.
if (${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(c10_opencl PRIVATE "-fvisibility=hidden")
endif()

# ---[ Dependency of c10_opencl
target_link_libraries(c10_opencl PUBLIC c10)

target_link_libraries(c10_opencl PUBLIC OpenCL)

target_include_directories(
    c10_opencl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

# ---[ Installation
# Note: for now, we will put all export path into one single Caffe2Targets group
# to deal with the cmake deployment need. Inside the Caffe2Targets set, the
# individual libraries like libc10.so and libcaffe2.so are still self-contained.
install(TARGETS c10_opencl EXPORT Caffe2Targets DESTINATION lib)
foreach(file ${C10_OPENCL_HEADERS})
  get_filename_component( dir ${file} DIRECTORY )
  file(RELATIVE_PATH dir_path ${CMAKE_SOURCE_DIR} ${dir})
  install( FILES ${file} DESTINATION include/${dir_path} )
endforeach()
install(FILES ${CMAKE_BINARY_DIR}/c10/opencl/impl/opencl_cmake_macros.h
  DESTINATION include/c10/opencl/impl)

if (MSVC AND C10_OPENCL_BUILD_SHARED_LIBS)
  install(FILES $<TARGET_PDB_FILE:c10_opencl> DESTINATION lib OPTIONAL)
endif()

set(_OPENCL_KERNEL_DIR ${_OPENCL_KERNEL_DIR} PARENT_SCOPE)
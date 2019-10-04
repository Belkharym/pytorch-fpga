# Build file for C10 OpenCL
#
# C10 OpenCL is a minimal library, but it does depend on OpenCL.

include(../../cmake/public/utils.cmake)

# ---[ Configure macro file.
set(C10_OPENCL_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/impl/opencl_cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/opencl/impl/opencl_cmake_macros.h)

# NB: All previous cu files are renamed into cc files.  This isn't tested at the
# moment.
file(GLOB C10_OPENCL_SRCS
    *.cpp
    impl/*.cpp
    )

message(INFO "C10_OPENCL_SRCS ${C10_OPENCL_SRCS}")

file(GLOB C10_OPENCL_HEADERS
    *.h
    impl/*.h
)

message(INFO "C10_OPENCL_HEADERS ${C10_OPENCL_HEADERS}")

add_library(c10_opencl ${C10_OPENCL_SRCS} ${C10_OPENCL_HEADERS})

# ---[ Dependency of c10_opencl
target_link_libraries(c10_opencl PUBLIC c10)

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
  install( FILES ${file} DESTINATION include/c10/opencl/${dir} )
endforeach()
install(FILES ${CMAKE_BINARY_DIR}/c10/opencl/impl/opencl_cmake_macros.h
  DESTINATION include/c10/opencl/impl)

if (MSVC AND C10_OPENCL_BUILD_SHARED_LIBS)
  install(FILES $<TARGET_PDB_FILE:c10_opencl> DESTINATION lib OPTIONAL)
endif()

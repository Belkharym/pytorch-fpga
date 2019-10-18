
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../cmake/Modules)

set(USE_CUDA 0)
set(USE_OPENCL ON)
set(USE_FPGA ON)

if (NOT SDACCEL_FOUND)
    find_package(SDAccel REQUIRED)
    if (NOT SDACCEL_FOUND)
        MESSAGE(FATAL_ERROR
            "Could not find SDAccel installation")
    endif()
endif()
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER ${SDACCEL_XCPP})
set(HAVE_STD_REGEX 0) # set to success, which is 0
message(STATUS "SDACCEL_LIBRARIES \"${SDACCEL_LIBRARIES}\"")
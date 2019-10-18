
if (USE_FPGA)
    find_package(SDAccel REQUIRED)
    if (NOT SDACCEL_FOUND)
        MESSAGE(FATAL_ERROR
            "Could not find SDAccel installation")
    endif()
    set(CMAKE_C_COMPILER ${SDACCEL_XCPP})
    set(CMAKE_CXX_COMPILER ${SDACCEL_XCPP})
endif()
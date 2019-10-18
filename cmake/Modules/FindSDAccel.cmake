# Author:  Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
# Created: October 2016
#
# To specify the path to the SDAccel installation, provide:
#   -DSDACCEL_ROOT_DIR=<installation directory>
# If successful, this script defines:
#   SDACCEL_FOUND
#   SDACCEL_INCLUDE_DIRS
#   SDACCEL_LIBRARIES
#   SDACCEL_XOCC
#   SDACCEL_VIVADO_HLS

# See https://github.com/definelicht/sdaccel_example

cmake_minimum_required(VERSION 2.8.12)

# set(SDACCEL_ROOT_DIR "/opt/Xilinx/SDx/2018.3.op2405991/")

find_path(XOCC_PATH
  NAMES xocc
  PATHS ${SDACCEL_ROOT_DIR} ENV XILINX_SDACCEL
  PATH_SUFFIXES bin/
)

if(NOT EXISTS ${XOCC_PATH})

  message(WARNING "SDAccel not found.")

else()

  get_filename_component(SDACCEL_ROOT_DIR ${XOCC_PATH} DIRECTORY)

  set(SDACCEL_FOUND TRUE)
  set(SDACCEL_INCLUDE_DIRS ${SDACCEL_ROOT_DIR}/include/)
  set(SDACCEL_XOCC ${SDACCEL_ROOT_DIR}/bin/xocc)
  set(SDACCEL_XCPP ${SDACCEL_ROOT_DIR}/bin/xcpp)
  set(SDACCEL_VIVADO_HLS ${SDACCEL_ROOT_DIR}/Vivado_HLS/bin/vivado_hls)

  # Runtime includes
  file(GLOB SUBDIRECTORIES ${SDACCEL_ROOT_DIR}/runtime/include/   
       ${SDACCEL_ROOT_DIR}/runtime/include/*)
  foreach(SUBDIRECTORY ${SUBDIRECTORIES})
    if(IS_DIRECTORY ${SUBDIRECTORY})
      set(SDACCEL_INCLUDE_DIRS ${SDACCEL_INCLUDE_DIRS} ${SUBDIRECTORY})
    endif()
  endforeach()

  # OpenCL libraries
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(SDACCEL_LIBRARY_DIR ${SDACCEL_ROOT_DIR}/runtime/lib/x86_64/)
    file(GLOB SDACCEL_LIBRARIES ${SDACCEL_ROOT_DIR}/runtime/lib/x86_64/
         ${SDACCEL_ROOT_DIR}/runtime/lib/x86_64/*.so)
  endif()

  message(STATUS "Found SDAccel at ${SDACCEL_ROOT_DIR}.")

endif()

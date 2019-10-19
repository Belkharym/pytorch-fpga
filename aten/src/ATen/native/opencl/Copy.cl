#include "aten/src/ATen/native/opencl/OpenCLKernelMacros.clh"
#include "aten/src/ATen/native/opencl/OpenCLOperations.h"

#define CAST_CASE_(name1, t2, _) \
  case name1: { \
    switch(t2) { \
      _(bool, BOOL) \
      _(char, CHAR) \
      _(short, SHORT) \
      _(int, INT) \
      _(long, LONG) \
      _(float, FLOAT) \
      _(double, DOUBLE) \
    } \
    break; \
  }

#define CAST_CASE(type2, name2) \
  case name2: { \
    ((__global type2*)b)[get_global_id(0)] = ((__global type2*)a)[get_global_id(0)]; \
    break; \
  }

__kernel void cast(__global const void *a, __global void *b, const enum OpenCLCastType t1, const enum OpenCLCastType t2) {
  switch(t1) {
    CAST_CASE_(BOOL, t2, CAST_CASE)
    CAST_CASE_(CHAR, t2, CAST_CASE)
    CAST_CASE_(SHORT, t2, CAST_CASE)
    CAST_CASE_(INT, t2, CAST_CASE)
    CAST_CASE_(LONG, t2, CAST_CASE)
    CAST_CASE_(FLOAT, t2, CAST_CASE)
    CAST_CASE_(DOUBLE, t2, CAST_CASE)
  }
}

#define CAST_CASE_S(type2, name2) \
  case name2: { \
    ((__global type2*)b)[get_global_id(0)] = *((__global type2*) a); \
    break; \
  }

// Scalar version
__kernel void cast_s(__global const void *a, __global void *b, const enum OpenCLCastType tb) {
  switch(tb) {
    CAST_CASE_S(bool, BOOL)
    CAST_CASE_S(char, CHAR)
    CAST_CASE_S(short, SHORT)
    CAST_CASE_S(int, INT)
    CAST_CASE_S(long, LONG)
    CAST_CASE_S(float, FLOAT)
    CAST_CASE_S(double, DOUBLE)
  }
}

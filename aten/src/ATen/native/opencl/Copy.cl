#include "aten/src/ATen/native/opencl/OpenCLKernelMacros.clh"
#include "aten/src/ATen/native/opencl/OpenCLOperations.h"

#define CAST_KERNEL(suffix1, type1, suffix2, type2)\
void cast_##suffix1##_##suffix2(__global const type1 *a, __global type2 *b) { \
  b[get_global_id(0)] = (type2)a[get_global_id(0)]; \
}

#define CAST_TO_ALL_TYPES(suffix, type) \
    CAST_KERNEL(suffix, type, b, bool) \
    CAST_KERNEL(suffix, type, c, char) \
    CAST_KERNEL(suffix, type, s, short) \
    CAST_KERNEL(suffix, type, i, int) \
    CAST_KERNEL(suffix, type, l, long) \
    CAST_KERNEL(suffix, type, f, float) \
    CAST_KERNEL(suffix, type, d, double)

CAST_TO_ALL_TYPES(b, bool)
DEFINE_KERNEL_FOR_ALL_TYPES(CAST_TO_ALL_TYPES)

#define CAST_CASE_(suffix1, type1, name1, t2, _) \
  case name1: { \
    switch(t2) { \
      _(suffix1, type1, b, bool, BOOL) \
      _(suffix1, type1, c, char, CHAR) \
      _(suffix1, type1, s, short, SHORT) \
      _(suffix1, type1, i, int, INT) \
      _(suffix1, type1, l, long, LONG) \
      _(suffix1, type1, f, float, FLOAT) \
      _(suffix1, type1, d, double, DOUBLE) \
    } \
    break; \
  }

#define CAST_CASE(suffix1, type1, suffix2, type2, name2) \
  case name2: { \
    cast_##suffix1##_##suffix2((__global const type1*)a, (__global type2*)b); \
    break; \
  }

__kernel void cast(__global const void *a, __global void *b, const enum OpenCLCastType t1, const enum OpenCLCastType t2) {
  switch(t1) {
    CAST_CASE_(b, bool, BOOL, t2, CAST_CASE)
    CAST_CASE_(c, char, CHAR, t2, CAST_CASE)
    CAST_CASE_(s, short, SHORT, t2, CAST_CASE)
    CAST_CASE_(i, int, INT, t2, CAST_CASE)
    CAST_CASE_(l, long, LONG, t2, CAST_CASE)
    CAST_CASE_(f, float, FLOAT, t2, CAST_CASE)
    CAST_CASE_(d, double, DOUBLE, t2, CAST_CASE)
  }
}

#define CAST_CASE_S(type1, type2, name2) \
  case name2: { \
    ((__global type2*)b)[get_global_id(0)] = (type2)a; \
    break; \
  }

// Scalar version
#define CAST_KERNEL_S(suffix1, type1)\
__kernel void cast_##suffix1##s(const type1 a, __global void *b, const enum OpenCLCastType tb) { \
  switch(tb) { \
    CAST_CASE_S(type1, bool, BOOL) \
    CAST_CASE_S(type1, char, CHAR) \
    CAST_CASE_S(type1, short, SHORT) \
    CAST_CASE_S(type1, int, INT) \
    CAST_CASE_S(type1, long, LONG) \
    CAST_CASE_S(type1, float, FLOAT) \
    CAST_CASE_S(type1, double, DOUBLE) \
  } \
}

DEFINE_KERNEL_FOR_ALL_TYPES(CAST_KERNEL_S)

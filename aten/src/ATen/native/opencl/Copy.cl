#include "aten/src/ATen/native/opencl/OpenCLKernelMacros.clh"

#define CAST_KERNEL(suffix1, type1, suffix2, type2)\
__kernel void cast_##suffix1##_##suffix2(__global const type1 *a, __global type2 *b) { \
  b[get_global_id(0)] = (type2)a[get_global_id(0)]; \
}

#define CAST_TO_ALL_TYPES(suffix, type) \
    CAST_KERNEL(suffix, type, b, bool) \
    CAST_KERNEL(suffix, type, c, char) \
    CAST_KERNEL(suffix, type, s, short) \
    CAST_KERNEL(suffix, type, i, int) \
    CAST_KERNEL(suffix, type, l, long) \
    CAST_KERNEL(suffix, type, f, float) \
    CAST_KERNEL(suffix, type, d, double) \

DEFINE_KERNEL_FOR_ALL_TYPES(CAST_TO_ALL_TYPES)

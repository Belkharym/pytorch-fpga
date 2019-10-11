#include <aten/src/ATen/native/opencl/MacroOpenCL.h>
#include "aten/src/ATen/native/opencl/OpenCLOperations.h"


// DECLARE_INT_COMP(half, int, h)
DECLARE_INT_COMP(float, int, f)
DECLARE_INT_COMP(double, long, d)
DECLARE_INT_COMP(char, char, c)
DECLARE_INT_COMP(short, short, s)
DECLARE_INT_COMP(int, int, i)
DECLARE_INT_COMP(long, long, l)

#define OPERATION(suffix, type) \
__kernel void operation_##suffix(__global const type* a, __global const type* other, __global type* out, __global const type alpha, enum OpenCLOperationsPointwise op) { \
    switch(op) { \
        case ADD: { \
            out[get_global_id(0)] = a[get_global_id(0)] + other[get_global_id(0)] * alpha; \
            break; \
        } \
    } \
}

DEFINE_KERNEL_FOR_ALL_TYPES(OPERATION)
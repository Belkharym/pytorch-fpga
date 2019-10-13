#include <aten/src/ATen/native/opencl/MacroOpenCL.h>
#include "aten/src/ATen/native/opencl/OpenCLOperations.h"


#define POINTWISE_OP_3S(suffix, type) \
__kernel void pointwise_op3##suffix##_s(__global const type* a, __global const type* other, __global type* out, const type alpha, const enum OpenCLOperationsPointwise3s op) { \
    switch(op) { \
        case ADD: { \
            out[get_global_id(0)] = a[get_global_id(0)] + other[get_global_id(0)] * alpha; \
            break; \
        } \
        case SUB: { \
            out[get_global_id(0)] = a[get_global_id(0)] - other[get_global_id(0)] * alpha; \
            break; \
        } \
        case MUL: { \
            out[get_global_id(0)] = a[get_global_id(0)] * other[get_global_id(0)] * alpha; \
            break; \
        } \
        case DIV: { \
            out[get_global_id(0)] = a[get_global_id(0)] / other[get_global_id(0)] * alpha; \
            break; \
        } \
    } \
}

DEFINE_KERNEL_FOR_ALL_TYPES(POINTWISE_OP_3S)

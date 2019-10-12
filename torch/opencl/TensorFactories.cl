// #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#include "aten/src/ATen/native/opencl/OpenCLOperations.h"
#include "aten/src/ATen/native/opencl/MacroOpenCL.h"

/// Code Declatation

__kernel void test() {
    (void)0;
}

#define DECLARE_INT_COMP(type, rettype, suffix) \
inline __attribute__((overloadable,always_inline)) rettype comp##suffix(const type a, const type b, enum OpenCLOperationsPointwise3 op) { \
    switch(op) {            \
        case EQ:            \
            return a == b;  \
            break;          \
        case NE:            \
            return a != b;  \
            break;          \
        case GT:            \
            return a > b;   \
            break;          \
        case LT:            \
            return a < b;   \
            break;          \
        case GE:            \
            return a >= b;  \
            break;          \
        case LE:            \
            return a <= b;  \
            break;          \
    }                       \
}

// DECLARE_INT_COMP(half, int, h)
DECLARE_INT_COMP(float, int, f)
DECLARE_INT_COMP(double, long, d)
DECLARE_INT_COMP(char, char, c)
DECLARE_INT_COMP(short, short, s)
DECLARE_INT_COMP(int, int, i)
DECLARE_INT_COMP(long, long, l)

// DONE See aten/src/THC/THCApply.cuh in the function THC_pointwiseApply3 for implementation details.
// The type 'Operations' is an enumaration, since we can't give a function pointer as a parameter
// in some implementations (for instance, on FPGAs). Thus we use predefined operations and we choose
// the operation within the enumaration (maybe using a switch-case statement).
#define POINTWISE_OP_3(suffix, type) \
__kernel void pointwise_op_3##suffix(__global const type* a, __global const type* b, __global type* c, enum OpenCLOperationsPointwise3 op) { \
    c[get_global_id(0)] = (type)comp##suffix((type)a[get_global_id(0)], (type)b[get_global_id(0)], op); \
}
DEFINE_KERNEL_FOR_ALL_TYPES(POINTWISE_OP_3)
#undef POINTWISE_OP_3

// Tensor and Scalar
#define POINTWISE_OP_2S(suffix, type) \
__kernel void pointwise_op_2##suffix##_s(__global const type* a, const type b, __global type* c, enum OpenCLOperationsPointwise3 op) { \
    c[get_global_id(0)] = comp##suffix(a[get_global_id(0)], b, op); \
}
DEFINE_KERNEL_FOR_ALL_TYPES(POINTWISE_OP_2S)
#undef POINTWISE_OP_2S

#define POINTWISE_OP_INT(suffix, type) \
__kernel void pointwise_op_##suffix(__global const type* a, __global type* b, enum OpenCLOperationsPointwise op) { \
    switch(op) { \
        case ABS: { \
            b[get_global_id(0)] = a[get_global_id(0)] < (type)0 ? -a[get_global_id(0)] : a[get_global_id(0)]; \
            break; \
        } \
        case CEIL: { \
            b[get_global_id(0)] = a[get_global_id(0)]; \
        } \
    } \
}

#define POINTWISE_OP_FLOAT(suffix, type) \
__kernel void pointwise_op_##suffix(__global const type* a, __global type* b, enum OpenCLOperationsPointwise op) { \
    switch(op) { \
        case ABS: { \
            b[get_global_id(0)] = a[get_global_id(0)] < (type)0 ? -a[get_global_id(0)] : a[get_global_id(0)]; \
            break; \
        } \
        case CEIL: { \
            b[get_global_id(0)] = ceil(a[get_global_id(0)]); \
        } \
    } \
}

DEFINE_KERNEL_FOR_INTS(POINTWISE_OP_INT)
DEFINE_KERNEL_FOR_FLOATS(POINTWISE_OP_FLOAT)
//DEFINE_KERNEL_FOR_ALL_TYPES(POINTWISE_OP)
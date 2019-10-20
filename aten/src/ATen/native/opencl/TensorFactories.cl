#include "aten/src/ATen/native/opencl/OpenCLKernelMacros.clh"
#include "aten/src/ATen/native/opencl/OpenCLOperations.h"

/// Code Declatation

#define DECLARE_INT_COMP(type, rettype, suffix) \
inline __attribute__((overloadable,always_inline)) rettype comp##suffix(const type a, const type b, const enum OpenCLOperationsComp3 op) { \
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

#define DECLARE_FP_COMP(type, rettype, suffix) \
inline __attribute__((overloadable,always_inline)) rettype comp##suffix(const type a, const type b, const enum OpenCLOperationsComp3 op) { \
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

DECLARE_FP_COMP(float, int, f)
DECLARE_FP_COMP(double, long, d)
DECLARE_INT_COMP(char, char, c)
DECLARE_INT_COMP(short, short, s)
DECLARE_INT_COMP(int, int, i)
DECLARE_INT_COMP(long, long, l)

// DONE See aten/src/THC/THCApply.cuh in the function THC_pointwiseApply3 for implementation details.
// The type 'Operations' is an enumaration, since we can't give a function pointer as a parameter
// in some implementations (for instance, on FPGAs). Thus we use predefined operations and we choose
// the operation within the enumaration (maybe using a switch-case statement).


#define POINTWISE_OP_COMP_3(suffix, type) \
    ((__global type*)out)[get_global_id(0)] = (__global type*)comp##suffix(((__global type*)a)[get_global_id(0)], ((__global type*)b)[get_global_id(0)], op);



#define POINTWISE_OP_COMP_3_CASE_(suffix, type, name, _) \
    case name: { \
        _(suffix, type) \
    break; \
  }



__kernel void pointwise_op_comp_3(__global const void* a, __global const void* b, __global void* out, const enum OpenCLOperationsComp3 op, const enum OpenCLCastType typeTensor) { \
    switch(typeTensor) {
        POINTWISE_OP_3_CASE_(b, bool, BOOL, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(c, char, CHAR, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(s, short, SHORT, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(i, int, INT, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(l, long, LONG, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(f, float, FLOAT, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(d, double, DOUBLE, POINTWISE_OP_3)
    }
}

#undef POINTWISE_OP_COMP_3
#undef POINTWISE_OP_COMP_3_CASE_

#define POINTWISE_OP_3_INT(suffix, type) \
__kernel void pointwise_op_3##suffix(__global const type* a, __global const type* b, __global type* out, const enum OpenCLOperationsPointwise3 op) { \
    switch(op) {                                                                                                            \
        case BAND: {                                                                                                        \
            out[get_global_id(0)] = a[get_global_id(0)] & b[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case MIN: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] < b[get_global_id(0)] ? a[get_global_id(0)] : b[get_global_id(0)];  \
            break;                                                                                                          \
        }                                                                                                                   \
        case MAX: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] > b[get_global_id(0)] ? a[get_global_id(0)] : b[get_global_id(0)];  \
            break;                                                                                                          \
        }                                                                                                                   \
        case ADD: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] + b[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case SUB: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] - b[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case MUL: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] * b[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case DIV: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] / b[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case REM: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] % b[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case BXOR: {                                                                                                        \
            out[get_global_id(0)] = a[get_global_id(0)] ^ b[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case ATAN2: {                                                                                                       \
            out[get_global_id(0)] = atan2((float)a[get_global_id(0)], (float)b[get_global_id(0)]);                          \
            break;                                                                                                          \
        }                                                                                                                   \
    }                                                                                                                       \
}
DEFINE_KERNEL_FOR_INTS(POINTWISE_OP_3_INT)
#undef POINTWISE_OP_3_INT

#define POINTWISE_OP_3_FP(suffix, type) \
__kernel void pointwise_op_3##suffix(__global const type* a, __global const type* b, __global type* out, const enum OpenCLOperationsPointwise3 op) { \
    switch(op) {                                                                                                            \
        case BAND: {                                                                                                        \
            out[get_global_id(0)] = (type)((bool)a[get_global_id(0)] && (bool)b[get_global_id(0)]);                         \
            break;                                                                                                          \
        }                                                                                                                   \
        case MIN: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] < b[get_global_id(0)] ? a[get_global_id(0)] : b[get_global_id(0)];  \
            break;                                                                                                          \
        }                                                                                                                   \
        case MAX: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] > b[get_global_id(0)] ? a[get_global_id(0)] : b[get_global_id(0)];  \
            break;                                                                                                          \
        }                                                                                                                   \
        case ADD: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] + b[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case SUB: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] - b[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case MUL: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] * b[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case DIV: {                                                                                                         \
            out[get_global_id(0)] = a[get_global_id(0)] / b[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case REM: {                                                                                                         \
            out[get_global_id(0)] = fmod((float)a[get_global_id(0)], (float)b[get_global_id(0)]);                         \
            break;                                                                                                          \
        }                                                                                                                   \
        case BXOR: {                                                                                                        \
            out[get_global_id(0)] = a[get_global_id(0)] != b[get_global_id(0)];                                             \
            break;                                                                                                          \
        }                                                                                                                   \
        case ATAN2: {                                                                                                       \
            out[get_global_id(0)] = atan2((float)a[get_global_id(0)], (float)b[get_global_id(0)]);                          \
            break;                                                                                                          \
        }                                                                                                                   \
    }                                                                                                                       \
}

DEFINE_KERNEL_FOR_FLOATS(POINTWISE_OP_3_FP)
#undef POINTWISE_OP_3_FP

// Tensor and Scalar
#define POINTWISE_OP_COMP_2S(suffix, type) \
__kernel void pointwise_op_comp_2##suffix##_s(__global const type* a, const type b, __global type* out, const enum OpenCLOperationsComp3 op) { \
    out[get_global_id(0)] = comp##suffix(a[get_global_id(0)], b, op); \
}
DEFINE_KERNEL_FOR_ALL_TYPES(POINTWISE_OP_COMP_2S)
#undef POINTWISE_OP_COMP_2S

#define POINTWISE_OP_2S_INT(suffix, type) \
__kernel void pointwise_op_2##suffix##_s(__global const type* a, const type b, __global type* out, const enum OpenCLOperationsPointwise3 op) { \
    switch(op) {                                                                        \
        case BAND: {                                                                    \
            out[get_global_id(0)] = a[get_global_id(0)] & b;                            \
            break;                                                                      \
        }                                                                               \
        case MIN: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] < b ? a[get_global_id(0)] : b;  \
            break;                                                                      \
        }                                                                               \
        case MAX: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] > b ? a[get_global_id(0)] : b;  \
            break;                                                                      \
        }                                                                               \
        case ADD: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] + b;                            \
            break;                                                                      \
        }                                                                               \
        case SUB: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] - b;                            \
            break;                                                                      \
        }                                                                               \
        case MUL: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] * b;                            \
            break;                                                                      \
        }                                                                               \
        case DIV: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] / b;                            \
            break;                                                                      \
        }                                                                               \
        case REM: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] % b;                            \
            break;                                                                      \
        }                                                                               \
        case BXOR: {                                                                    \
            out[get_global_id(0)] = a[get_global_id(0)] ^ b;                            \
            break;                                                                      \
        }                                                                               \
        case ATAN2: {                                                                   \
            out[get_global_id(0)] = atan2((float)a[get_global_id(0)], (float)b);        \
            break;                                                                      \
        }                                                                               \
    }                                                                                   \
}
DEFINE_KERNEL_FOR_INTS(POINTWISE_OP_2S_INT)
#undef POINTWISE_OP_2S_INT

#define POINTWISE_OP_2S_FP(suffix, type) \
__kernel void pointwise_op_2##suffix##_s(__global const type* a, const type b, __global type* out, const enum OpenCLOperationsPointwise3 op) { \
    switch(op) {                                                                        \
        case BAND: {                                                                    \
            out[get_global_id(0)] = (type)((bool)a[get_global_id(0)] && (bool)b);       \
            break;                                                                      \
        }                                                                               \
        case MIN: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] < b ? a[get_global_id(0)] : b;  \
            break;                                                                      \
        }                                                                               \
        case MAX: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] > b ? a[get_global_id(0)] : b;  \
            break;                                                                      \
        }                                                                               \
        case ADD: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] + b;                            \
            break;                                                                      \
        }                                                                               \
        case SUB: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] - b;                            \
            break;                                                                      \
        }                                                                               \
        case MUL: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] * b;                            \
            break;                                                                      \
        }                                                                               \
        case DIV: {                                                                     \
            out[get_global_id(0)] = a[get_global_id(0)] / b;                            \
            break;                                                                      \
        }                                                                               \
        case REM: {                                                                     \
            out[get_global_id(0)] = fmod((float)a[get_global_id(0)], (float)b);         \
            break;                                                                      \
        }                                                                               \
        case BXOR: {                                                                    \
            out[get_global_id(0)] = a[get_global_id(0)] != b;                           \
            break;                                                                      \
        }                                                                               \
        case ATAN2: {                                                                   \
            out[get_global_id(0)] = atan2((float)a[get_global_id(0)], (float)b);        \
            break;                                                                      \
        }                                                                               \
    }                                                                                   \
}
DEFINE_KERNEL_FOR_FLOATS(POINTWISE_OP_2S_FP)
#undef POINTWISE_OP_2S_FP

#define POINTWISE_OP_2_INT(suffix, type) \
__kernel void pointwise_op_2##suffix(__global const type* a, __global type* out, const enum OpenCLOperationsPointwise2 op) { \
    switch(op) { \
        case ABS: { \
            out[get_global_id(0)] = a[get_global_id(0)] < (type)0 ? -a[get_global_id(0)] : a[get_global_id(0)]; \
            break; \
        } \
        case CEIL: { \
            out[get_global_id(0)] = a[get_global_id(0)]; \
            break; \
        } \
    } \
}


DEFINE_KERNEL_FOR_INTS(POINTWISE_OP_2_INT)

#undef POINTWISE_OP_2_INT


#define POINTWISE_OP_2_FLOAT(suffix, type) \
__kernel void pointwise_op_2##suffix(__global const type* a, __global type* out, const enum OpenCLOperationsPointwise2 op) { \
    switch(op) { \
        case ABS: { \
            out[get_global_id(0)] = a[get_global_id(0)] < (type)0 ? -a[get_global_id(0)] : a[get_global_id(0)]; \
            break; \
        } \
        case CEIL: { \
            out[get_global_id(0)] = ceil((type)a[get_global_id(0)]); \
            break; \
        } \
    } \
}

POINTWISE_OP_2_FLOAT(f, float)
POINTWISE_OP_2_FLOAT(d, float)
//DEFINE_KERNEL_FOR_ALL_TYPES(POINTWISE_OP_2)
#undef POINTWISE_OP_2_FLOAT


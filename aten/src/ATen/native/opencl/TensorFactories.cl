#include "aten/src/ATen/native/opencl/OpenCLKernelMacros.clh"
#include "aten/src/ATen/native/opencl/OpenCLOperations.h"
#include "aten/src/ATen/native/opencl/MacroOpenCL.h"

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



__kernel void pointwise_op_comp_3(__global const void* a, __global const void* b, __global void* out, const enum OpenCLOperationsComp3 op, const enum OpenCLCastType typeTensor) {
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

#define POINTWISE_REM_FLOAT(type) \
    ((__global type*)out)[get_global_id(0)] = fmod(((__global typ*)a)[get_global_id(0)], ((__global type*)b)[get_global_id(0)]);

#define POINTWISE_REM_INT(type) \
    ((__global type*)out)[get_global_id(0)] = ((__global typ*)a)[get_global_id(0)] % ((__global type*)b)[get_global_id(0)];

#define REM_CASE(type, name, _) \
    case name: { \
        _(type) \
        break; \
    }

#define POINTWISE_OP_3(type, name) \
    switch(op) {                                                                                                            \
        case BAND: {                                                                                                        \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] & ((__global type*)b)[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case MIN: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] < ((__global type*)b)[get_global_id(0)] ? ((__global type*)a)[get_global_id(0)] : ((__global type*)b)[get_global_id(0)];  \
            break;                                                                                                          \
        }                                                                                                                   \
        case MAX: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] > ((__global type*)b)[get_global_id(0)] ? ((__global type*)a)[get_global_id(0)] : ((__global type*)b)[get_global_id(0)];  \
            break;                                                                                                          \
        }                                                                                                                   \
        case ADD: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] + ((__global type*)b)[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case SUB: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] - ((__global type*)b)[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case MUL: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] * ((__global type*)b)[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case DIV: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] / ((__global type*)b)[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case REM: {                                                                                                         \
            switch(typeTensor) {                                                                                            \
                REM_CASE(bool, BOOL, POINTWISE_REM_INT)                                                                     \
                REM_CASE(char, CHAR, POINTWISE_REM_INT)                                                                     \
                REM_CASE(short, SHORT, POINTWISE_REM_INT)                                                                   \
                REM_CASE(int, INT, POINTWISE_REM_INT)                                                                       \
                REM_CASE(long, LONG, POINTWISE_REM_INT)                                                                     \
                REM_CASE(float, FLOAT, POINTWISE_REM_FLOAT)                                                                 \
                REM_CASE(double, DOUBLE, POINTWISE_REM_FLOAT)                                                               \
            }                                                                                                               \
            break;                                                                                                          \
        }                                                                                                                   \
        case BXOR: {                                                                                                        \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] ^ ((__global type*)b)[get_global_id(0)];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case ATAN2: {                                                                                                       \
            out[get_global_id(0)] = atan2((float)a[get_global_id(0)], (float)b[get_global_id(0)]);                          \
            break;                                                                                                          \
        }                                                                                                                   \
    }            
}

#define POINTWISE_OP_3_CASE_(type, name, _) \
    case name: { \
        _(type, name) \
        break; \
    }


__kernel void pointwise_op_3(__global const void* a, __global const void* b, __global void* out, const enum OpenCLOperationsPointwise3 op, const enum OpenCLCastType typeTensor) { 
    switch(typeTensor) {
        POINTWISE_OP_3_CASE_(bool, BOOL, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(char, CHAR, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(short, SHORT, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(int, INT, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(long, LONG, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(float, FLOAT, POINTWISE_OP_3)
        POINTWISE_OP_3_CASE_(double, DOUBLE, POINTWISE_OP_3)
    }
    
}

#undef POINTWISE_REM_FLOAT
#undef POINTWISE_REM_INT
#undef REM_CASE
#undef POINTWISE_OP_3
#undef POINTWISE_OP_3_CASE_

// Tensor and Scalar

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
        default:{ \
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
        default:{ \
            break; \
        } \
    } \
}

POINTWISE_OP_2_FLOAT(f, float)
POINTWISE_OP_2_FLOAT(d, float)
//DEFINE_KERNEL_FOR_ALL_TYPES(POINTWISE_OP_2)
#undef POINTWISE_OP_2_FLOAT


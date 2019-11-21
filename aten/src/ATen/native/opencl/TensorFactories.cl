#include <OpenCLKernelMacros.clh>
#include <OpenCLOperations.h>

/// Code Declatation

#define DECLARE_INT_COMP(type, rettype, suffix) \
inline __attribute__((always_inline)) rettype comp##suffix(const type a, const type b, const enum OpenCLOperationsComp3 op) { \
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
inline __attribute__((always_inline)) rettype comp##suffix(const type a, const type b, const enum OpenCLOperationsComp3 op) { \
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
DECLARE_INT_COMP(char, char, b)
DECLARE_INT_COMP(char, char, c)
DECLARE_INT_COMP(short, short, s)
DECLARE_INT_COMP(int, int, i)
DECLARE_INT_COMP(long, long, l)

// DONE See aten/src/THC/THCApply.cuh in the function THC_pointwiseApply3 for implementation details.
// The type 'Operations' is an enumaration, since we can't give a function pointer as a parameter
// in some implementations (for instance, on FPGAs). Thus we use predefined operations and we choose
// the operation within the enumaration (maybe using a switch-case statement).


#define POINTWISE_OP_COMP_3(suffix, type) \
    ((__global type*)out)[get_global_id(0)] = (__global type)comp##suffix(((__global type*)a)[get_global_id(0)], ((__global type*)b)[get_global_id(0)], op);

#define POINTWISE_OP_COMP_2S(suffix, type) \
    ((__global type*)out)[get_global_id(0)] = (__global type)comp##suffix(((__global type*)a)[get_global_id(0)], ((__global type*)b)[0], op);

#define POINTWISE_OP_COMP_CASE_(suffix, type, name, _) \
    case name: { \
        _(suffix, type) \
    break; \
  }


__kernel void pointwise_op_comp_3(__global const void* a, __global const void* b, __global void* out, const enum OpenCLOperationsComp3 op, const enum OpenCLPtrType typeTensor) {
    switch(typeTensor) {
        POINTWISE_OP_COMP_CASE_(b, bool, BOOL, POINTWISE_OP_COMP_3)
        POINTWISE_OP_COMP_CASE_(c, char, CHAR, POINTWISE_OP_COMP_3)
        POINTWISE_OP_COMP_CASE_(s, short, SHORT, POINTWISE_OP_COMP_3)
        POINTWISE_OP_COMP_CASE_(i, int, INT, POINTWISE_OP_COMP_3)
        POINTWISE_OP_COMP_CASE_(l, long, LONG, POINTWISE_OP_COMP_3)
        POINTWISE_OP_COMP_CASE_(f, float, FLOAT, POINTWISE_OP_COMP_3)
        POINTWISE_OP_COMP_CASE_(d, double, DOUBLE, POINTWISE_OP_COMP_3)
        // case FLOAT: // passthrough
        // case DOUBLE: // passthrough
        //     break;
    }
}

__kernel void pointwise_op_comp_2s(__global const void* a, __global const void* b, __global void* out, const enum OpenCLOperationsComp3 op, const enum OpenCLPtrType typeTensor) {
    switch(typeTensor) {
        POINTWISE_OP_COMP_CASE_(b, bool, BOOL, POINTWISE_OP_COMP_2S)
        POINTWISE_OP_COMP_CASE_(c, char, CHAR, POINTWISE_OP_COMP_2S)
        POINTWISE_OP_COMP_CASE_(s, short, SHORT, POINTWISE_OP_COMP_2S)
        POINTWISE_OP_COMP_CASE_(i, int, INT, POINTWISE_OP_COMP_2S)
        POINTWISE_OP_COMP_CASE_(l, long, LONG, POINTWISE_OP_COMP_2S)
        POINTWISE_OP_COMP_CASE_(f, float, FLOAT, POINTWISE_OP_COMP_2S)
        POINTWISE_OP_COMP_CASE_(d, double, DOUBLE, POINTWISE_OP_COMP_2S)
        // case FLOAT: // passthrough
        // case DOUBLE: // passthrough
        //     break;
    }
}

#undef POINTWISE_OP_COMP_3
#undef POINTWISE_OP_COMP_3_CASE_

#define POINTWISE_REM_FLOAT(type) \
    ((__global type*)out)[get_global_id(0)] = fmod((float)((__global type*)a)[get_global_id(0)], (float)((__global type*)b)[get_global_id(0)]);

#define POINTWISE_REM_INT(type) \
    ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] % ((__global type*)b)[get_global_id(0)];

#define POINTWISE_2S_REM_FLOAT(type) \
    ((__global type*)out)[get_global_id(0)] = fmod((float)((__global type*)a)[get_global_id(0)], (float)((__global type*)b)[0]);

#define POINTWISE_2S_REM_INT(type) \
    ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] % ((__global type*)b)[0];

#define OP_CASE(type, name, _) \
    case name: { \
        _(type) \
        break; \
    }

#define POINTWISE_OP_3_INT(type, name) \
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
            POINTWISE_REM_INT(type)                                                                                         \
            break;                                                                                                          \
        }                                                                                                                   \
        case BXOR: {                                                                                                        \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] ^ ((__global type*)b)[get_global_id(0)]; \
            break;                                                                                                          \
        }                                                                                                                   \
        case ATAN2: {                                                                                                       \
            ((__global type*)out)[get_global_id(0)] = atan2((float)((__global type*)a)[get_global_id(0)], (float)((__global type*)b)[get_global_id(0)]);                          \
            break;                                                                                                          \
        }                                                                                                                   \
    }

#define POINTWISE_OP_3_FLOAT(type, name) \
    switch(op) {                                                                                                            \
        case BAND: {                                                                                                        \
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
            POINTWISE_REM_FLOAT(type)                                                                                       \
            break;                                                                                                          \
        }                                                                                                                   \
        case BXOR: {                                                                                                        \
            ((__global type*)out)[get_global_id(0)] = (bool)((__global type*)a)[get_global_id(0)] ^ (bool)((__global type*)b)[get_global_id(0)]; \
            break;                                                                                                          \
        }                                                                                                                   \
        case ATAN2: {                                                                                                       \
            ((__global type*)out)[get_global_id(0)] = (__global type)atan2((float)((__global type*)a)[get_global_id(0)], (float)((__global type*)b)[get_global_id(0)]);                          \
            break;                                                                                                          \
        }                                                                                                                   \
    }

#define POINTWISE_OP_2S_INT(type, name) \
    switch(op) {                                                                                                            \
        case BAND: {                                                                                                        \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] & ((__global type*)b)[0];       \
            break;                                                                                                          \
        }                                                                                                                   \
        case MIN: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] < ((__global type*)b)[0] ? ((__global type*)a)[get_global_id(0)] : ((__global type*)b)[0];  \
            break;                                                                                                          \
        }                                                                                                                   \
        case MAX: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] > ((__global type*)b)[0] ? ((__global type*)a)[get_global_id(0)] : ((__global type*)b)[0];  \
            break;                                                                                                          \
        }                                                                                                                   \
        case ADD: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] + ((__global type*)b)[0];       \
            break;                                                                                                          \
        }                                                                                                                   \
        case SUB: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] - ((__global type*)b)[0];       \
            break;                                                                                                          \
        }                                                                                                                   \
        case MUL: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] * ((__global type*)b)[0];       \
            break;                                                                                                          \
        }                                                                                                                   \
        case DIV: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] / ((__global type*)b)[0];       \
            break;                                                                                                          \
        }                                                                                                                   \
        case REM: {                                                                                                         \
            POINTWISE_2S_REM_INT(type)                                                                                      \
            break;                                                                                                          \
        }                                                                                                                   \
        case BXOR: {                                                                                                        \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] ^ ((__global type*)b)[0];       \
            break;                                                                                                          \
        }                                                                                                                   \
        case ATAN2: {                                                                                                       \
            ((__global type*)out)[get_global_id(0)] = (__global type)atan2((float)((__global type*)a)[get_global_id(0)], (float)((__global type*)b)[0]);    \
            break;                                                                                                          \
        }                                                                                                                   \
    }

#define POINTWISE_OP_2S_FLOAT(type, name) \
    switch(op) {                                                                                                            \
        case BAND: {                                                                                                        \
            break;                                                                                                          \
        }                                                                                                                   \
        case MIN: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] < ((__global type*)b)[0] ? ((__global type*)a)[get_global_id(0)] : ((__global type*)b)[0];  \
            break;                                                                                                          \
        }                                                                                                                   \
        case MAX: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] > ((__global type*)b)[0] ? ((__global type*)a)[get_global_id(0)] : ((__global type*)b)[0];  \
            break;                                                                                                          \
        }                                                                                                                   \
        case ADD: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] + ((__global type*)b)[0];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case SUB: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] - ((__global type*)b)[0];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case MUL: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] * ((__global type*)b)[0];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case DIV: {                                                                                                         \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] / ((__global type*)b)[0];                                              \
            break;                                                                                                          \
        }                                                                                                                   \
        case REM: {                                                                                                         \
            POINTWISE_2S_REM_FLOAT(type)                                                                                    \
            break;                                                                                                          \
        }                                                                                                                   \
        case BXOR: {                                                                                                        \
            ((__global type*)out)[get_global_id(0)] = (bool)((__global type*)a)[get_global_id(0)] ^ (bool)((__global type*)b)[0]; \
            break;                                                                                                          \
        }                                                                                                                   \
        case ATAN2: {                                                                                                       \
            ((__global type*)out)[get_global_id(0)] = (__global type)atan2((float)((__global type*)a)[get_global_id(0)], (float)((__global type*)b)[0]);                          \
            break;                                                                                                          \
        }                                                                                                                   \
    }

#define POINTWISE_OP_CASE_(type, name, _) \
    case name: { \
        _(type, name) \
        break; \
    }


__kernel void pointwise_op_3(__global const void* a, __global const void* b, __global void* out, const enum OpenCLOperationsPointwise3 op, const enum OpenCLPtrType typeTensor) {
    switch(typeTensor) {
        POINTWISE_OP_CASE_(bool, BOOL, POINTWISE_OP_3_INT)
        POINTWISE_OP_CASE_(char, CHAR, POINTWISE_OP_3_INT)
        POINTWISE_OP_CASE_(short, SHORT, POINTWISE_OP_3_INT)
        POINTWISE_OP_CASE_(int, INT, POINTWISE_OP_3_INT)
        POINTWISE_OP_CASE_(long, LONG, POINTWISE_OP_3_INT)
        POINTWISE_OP_CASE_(float, FLOAT, POINTWISE_OP_3_FLOAT)
        POINTWISE_OP_CASE_(double, DOUBLE, POINTWISE_OP_3_FLOAT)
        // case FLOAT: // passthrough
        // case DOUBLE: // passthrough
        //     break;
    }
}

__kernel void pointwise_op_2s(__global const void* a, __global const void* b, __global void* out, const enum OpenCLOperationsPointwise3 op, const enum OpenCLPtrType typeTensor) {
    switch(typeTensor) {
        POINTWISE_OP_CASE_(bool, BOOL, POINTWISE_OP_2S_INT)
        POINTWISE_OP_CASE_(char, CHAR, POINTWISE_OP_2S_INT)
        POINTWISE_OP_CASE_(short, SHORT, POINTWISE_OP_2S_INT)
        POINTWISE_OP_CASE_(int, INT, POINTWISE_OP_2S_INT)
        POINTWISE_OP_CASE_(long, LONG, POINTWISE_OP_2S_INT)
        POINTWISE_OP_CASE_(float, FLOAT, POINTWISE_OP_2S_FLOAT)
        POINTWISE_OP_CASE_(double, DOUBLE, POINTWISE_OP_2S_FLOAT)
        // case FLOAT: // passthrough
        // case DOUBLE: // passthrough
        //     break;
    }
}
#undef POINTWISE_REM_FLOAT
#undef POINTWISE_REM_INT
#undef POINTWISE_OP_3
#undef POINTWISE_2S_REM_FLOAT
#undef POINTWISE_2S_REM_INT
#undef POINTWISE_OP_2S

#define POINTWISE_CEIL_INT(type) \
     ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)];

#define POINTWISE_CEIL_FLOAT(type) \
     ((__global type*)out)[get_global_id(0)] = (float)ceil((float)((__global type*)a)[get_global_id(0)]);


#define POINTWISE_OP_2(type, name) \
    switch(op) { \
        case ABS: { \
            ((__global type*)out)[get_global_id(0)] = ((__global type*)a)[get_global_id(0)] < (type)0 ? -((__global type*)a)[get_global_id(0)] : ((__global type*)a)[get_global_id(0)]; \
            break; \
        } \
        case CEIL: { \
            switch (typeTensor) {                                                                                            \
                OP_CASE(bool, BOOL, POINTWISE_CEIL_INT)                                                                     \
                OP_CASE(char, CHAR, POINTWISE_CEIL_INT)                                                                     \
                OP_CASE(short, SHORT, POINTWISE_CEIL_INT)                                                                   \
                OP_CASE(int, INT, POINTWISE_CEIL_INT)                                                                       \
                OP_CASE(long, LONG, POINTWISE_CEIL_INT)                                                                     \
                OP_CASE(float, FLOAT, POINTWISE_CEIL_FLOAT)                                                                 \
                OP_CASE(double, DOUBLE, POINTWISE_CEIL_FLOAT)                                                               \
                /* case FLOAT: */ \
                /* case DOUBLE: */ \
                /*     break;  */\
            } \
            break; \
        } \
    }

__kernel void pointwise_op_2(__global const void* a, __global void* out, const enum OpenCLOperationsPointwise2 op, const enum OpenCLPtrType typeTensor) {
    switch(typeTensor) {
        POINTWISE_OP_CASE_(bool, BOOL, POINTWISE_OP_2)
        POINTWISE_OP_CASE_(char, CHAR, POINTWISE_OP_2)
        POINTWISE_OP_CASE_(short, SHORT, POINTWISE_OP_2)
        POINTWISE_OP_CASE_(int, INT, POINTWISE_OP_2)
        POINTWISE_OP_CASE_(long, LONG, POINTWISE_OP_2)
        POINTWISE_OP_CASE_(float, FLOAT, POINTWISE_OP_2)
        POINTWISE_OP_CASE_(double, DOUBLE, POINTWISE_OP_2)
        // case FLOAT: // passthrough
        // case DOUBLE: // passthrough
        //     break;
    }
}

#undef POINTWISE_CEIL_FLOAT
#undef POINTWISE_CEIL_INT
#undef POINTWISE_OP_CASE_
#undef OP_CASE
#undef POINTWISE_OP_2


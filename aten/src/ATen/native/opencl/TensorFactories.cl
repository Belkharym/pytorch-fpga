#include "aten/src/ATen/native/opencl/OpenCLOperations.h"

static float comp(const float a, const float b, enum OpenCLOperationsPointwise3 op) {
    switch(op) {
        case EQ:
            return a == b;
            break;
        case NE:
            return a != b;
            break;
        case GT:
            return a > b;
            break;
        case LT:
            return a < b;
            break;
        case GE:
            return a >= b;
            break;
        case LE:
            return a <= b;
            break;
    }
}

__kernel void pointwise_op_3f(__global const float* a, __global const float* b, __global float* c, enum OpenCLOperationsPointwise3 op) {
    // TODO See aten/src/THC/THCApply.cuh in the function THC_pointwiseApply3 for implementation details.
    // The type 'Operations' is an enumaration, since we can't give a function pointer as a parameter
    // in some implementations (for instance, on FPGAs). Thus we use predefined operations and we choose
    // the operation within the enumaration (maybe using a switch-case statement).
    c[get_global_id(0)] = comp(a[get_global_id(0)], b[get_global_id(0)], op);
}

__kernel void pointwise_op_2fv(__global const float* a, const float b, __global float* c, enum OpenCLOperationsPointwise3 op) {
    // TODO See aten/src/THC/THCApply.cuh in the function THC_pointwiseApply3 for implementation details.
    // The type 'Operations' is an enumaration, since we can't give a function pointer as a parameter
    // in some implementations (for instance, on FPGAs). Thus we use predefined operations and we choose
    // the operation within the enumaration (maybe using a switch-case statement).
    c[get_global_id(0)] = comp(a[get_global_id(0)], b, op);
}

__kernel void pointwise_op_f(__global const float* a, __global float* b, enum OpenCLOperationsPointwise op) {
    // TODO See aten/src/THC/THCApply.cuh in the function THC_pointwiseApply3 for implementation details.
    // The type 'Operations' is an enumaration, since we can't give a function pointer as a parameter
    // in some implementations (for instance, on FPGAs). Thus we use predefined operations and we choose
    // the operation within the enumaration (maybe using a switch-case statement).
    switch(op) {
        case ABS:
            b[get_global_id(0)] = a[get_global_id(0)] < 0.0f ? -a[get_global_id(0)] : a[get_global_id(0)];
            break;
    }
}


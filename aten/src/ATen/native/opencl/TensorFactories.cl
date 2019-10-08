#include "aten/src/ATen/native/opencl/OpenCLOperations.h"

__kernel void pointwise_op_3f(__global const float* a, __global const float* b, __global float* c, enum OpenCLOperations op) {
    // TODO See aten/src/THC/THCApply.cuh in the function THC_pointwiseApply3 for implementation details.
    // The type 'Operations' is an enumaration, since we can't give a function pointer as a parameter
    // in some implementations (for instance, on FPGAs). Thus we use predefined operations and we choose
    // the operation within the enumaration (maybe using a switch-case statement).
    switch(op) {
        case EQ:
            c[get_global_id(0)] = a[get_global_id(0)] == b[get_global_id(0)];
            break;
        case NE:
            c[get_global_id(0)] = a[get_global_id(0)] != b[get_global_id(0)];
            break;
        case GT:
            c[get_global_id(0)] = a[get_global_id(0)] > b[get_global_id(0)];
            break;
        case LT:
            c[get_global_id(0)] = a[get_global_id(0)] < b[get_global_id(0)];
            break;
        case GE:
            c[get_global_id(0)] = a[get_global_id(0)] >= b[get_global_id(0)];
            break;
        case LE:
            c[get_global_id(0)] = a[get_global_id(0)] <= b[get_global_id(0)];
            break;
    }
}

__kernel void pointwise_op_f(__global const float* a, __global float* c, enum OpenCLOperations op) {
    // TODO See aten/src/THC/THCApply.cuh in the function THC_pointwiseApply3 for implementation details.
    // The type 'Operations' is an enumaration, since we can't give a function pointer as a parameter
    // in some implementations (for instance, on FPGAs). Thus we use predefined operations and we choose
    // the operation within the enumaration (maybe using a switch-case statement).
    switch(op) {
        case ABS:
            b[get_global_id(0)] = abs(a[get_global_id(0)]);
            break;
    }
}


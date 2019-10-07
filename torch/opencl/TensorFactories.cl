#include "aten/src/ATen/native/opencl/OpenCLOperations.h"

__kernel void pointwise_op(__local long* a, __local long* b, __local const long* c, enum OpenCLOperations op) {
    // TODO See aten/src/THC/THCApply.cuh in the function THC_pointwiseApply3 for implementation details.
    // The type 'Operations' is an enumaration, since we can't give a function pointer as a parameter
    // in some implementations (for instance, on FPGAs). Thus we use predefined operations and we choose
    // the operation within the enumaration (maybe using a switch-case statement).
    (void)a;
    (void)b;
    (void)c;
    (void)op;
}
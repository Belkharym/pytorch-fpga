#ifndef OPENCL_OPERATORS_H
#define OPENCL_OPERATORS_H

#ifdef __cplusplus
namespace at {
namespace native {
namespace opencl {
#endif

enum OpenCLOperationsPointwise3 {
    // Comparisons
    EQ, // EQuals
    NE, // Not Equals
    GT, // Greater Than
    LT, // Less Than
    GE, // Greater than or Equal
    LE, // Less than or Equal
    // Bitwise operations
    BAND,
};

enum OpenCLOperationsPointwise {
    // Unary pointwise
    ABS, // ABSolute value
    CEIL, // Rounds x upward
    ADD, // addition
    SUB, // subtract
    DIV, // divide
    MUL, // multiply
};

#ifdef __cplusplus
}}} // namespace at::native::opencl
#endif

#endif // !OPENCL_OPERATORS_H
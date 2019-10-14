#ifndef OPENCL_OPERATORS_H
#define OPENCL_OPERATORS_H

#ifdef __cplusplus
namespace at {
namespace native {
namespace opencl {
#endif

// Logical operation
enum OpenCLOperationsComp3 {
    // Comparisons
    EQ, // EQuals
    NE, // Not Equals
    GT, // Greater Than
    LT, // Less Than
    GE, // Greater than or Equal
    LE, // Less than or Equal
};

// Operations that have 1 output and 2 inputs, all tensors of the same type
enum OpenCLOperationsPointwise3 {
    // Bitwise operations
    BAND, // Bitwise AND
    BXOR, // Bitwise XOR
    MUL, // multiply
    DIV, // divide
    ATAN2, // arc tangent of y / x.
};

// Operations that have 1 output and 1 input, all tensors of the same type
enum OpenCLOperationsPointwise2 {
    // Unary pointwise
    ABS, // ABSolute value
    CEIL, // Rounds x upward
};

// Operations that have 1 output tensor, 2 input tensors and 1 input scalar, all of the same type
enum OpenCLOperationsPointwise3s {
    // Algebraic operations
    ADD, // addition
    SUB, // subtract
};

#ifdef __cplusplus
}}} // namespace at::native::opencl
#endif

#endif // !OPENCL_OPERATORS_H

#ifndef OPENCL_OPERATORS_H
#define OPENCL_OPERATORS_H

#ifdef __cplusplus
namespace at {
namespace native {
namespace opencl {
#endif

enum OpenCLOperations {
    EQ, // EQuals
    NE, // Not Equals
    GT, // Greater Than
    LT, // Less Than
    GE, // Greater than or Equal
    LE, // Less than or Equal
};

#ifdef __cplusplus
}}} // namespace at::native::opencl
#endif

#endif // !OPENCL_OPERATORS_H

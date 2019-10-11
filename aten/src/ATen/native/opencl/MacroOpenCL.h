#ifndef MacroOpenCL_H
#define MacroOpenCL_H

#define DECLARE_FOR_ALL_SUBTYPES_EXCEPT_1(type, DECLARE_MACRO_) \
    DECLARE_MACRO_(type##2) \
    DECLARE_MACRO_(type##3) \
    DECLARE_MACRO_(type##4) \
    DECLARE_MACRO_(type##8) \
    DECLARE_MACRO_(type##16)

#define DECLARE_FOR_ALL_SUBTYPES(type, DECLARE_MACRO_) \
    DECLARE_MACRO_(type) \
    DECLARE_MACRO_(type##2) \
    DECLARE_MACRO_(type##3) \
    DECLARE_MACRO_(type##4) \
    DECLARE_MACRO_(type##8) \
    DECLARE_MACRO_(type##16)

#define DECLARE_FOR_ALL_SUBTYPES2_EXCEPT_1(type, type2, DECLARE_MACRO_) \
    DECLARE_MACRO_(type##2, type2##2) \
    DECLARE_MACRO_(type##3, type2##3) \
    DECLARE_MACRO_(type##4, type2##4) \
    DECLARE_MACRO_(type##8, type2##8) \
    DECLARE_MACRO_(type##16, type2##16)

#define DECLARE_FOR_ALL_SUBTYPES2(type, type2, DECLARE_MACRO_) \
    DECLARE_MACRO_(type, type2) \
    DECLARE_MACRO_(type##2, type2##2) \
    DECLARE_MACRO_(type##3, type2##3) \
    DECLARE_MACRO_(type##4, type2##4) \
    DECLARE_MACRO_(type##8, type2##8) \
    DECLARE_MACRO_(type##16, type2##16)

#define DEFINE_KERNEL_FOR_FLOATS(KERNEL_MACRO_) \
KERNEL_MACRO_(f, float)    \
KERNEL_MACRO_(d, double)

#define DEFINE_KERNEL_FOR_INTS(KERNEL_MACRO_) \
KERNEL_MACRO_(c, char)     \
KERNEL_MACRO_(s, short)    \
KERNEL_MACRO_(i, int)      \
KERNEL_MACRO_(l, long)

#define DEFINE_KERNEL_FOR_ALL_TYPES(KERNEL_MACRO_) \
DEFINE_KERNEL_FOR_INTS(KERNEL_MACRO_) \
DEFINE_KERNEL_FOR_FLOATS(KERNEL_MACRO_)

#endif
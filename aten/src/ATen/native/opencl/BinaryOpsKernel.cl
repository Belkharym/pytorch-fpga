#include <OpenCLKernelMacros.clh>
#include <OpenCLOperations.h>

#define OPERATION(type1, type2) \
switch(op) { \
  case ADDS: { \
    ((__global type1*)out)[get_global_id(0)] = ((__global type1*)a)[get_global_id(0)] + ((__global type1*)other)[get_global_id(0)] * ((__global type2*)alpha)[get_global_id(0)]; \
    break; \
  } \
  case SUBS: { \
    ((__global type1*)out)[get_global_id(0)] = ((__global type1*)a)[get_global_id(0)] - ((__global type1*)other)[get_global_id(0)] * ((__global type2*)alpha)[get_global_id(0)]; \
    break; \
  } \
}

#define OPERATION_CASE(type1, type2, name2) \
  case name2: { \
    OPERATION(type1, type2) \
    break; \
  }

#define OPERATION_CASE_(type1, name1, t2, _) \
  case name1: { \
    switch(t2) { \
      _(type1, bool, BOOL) \
      _(type1, char, CHAR) \
      _(type1, short, SHORT) \
      _(type1, int, INT) \
      _(type1, long, LONG) \
      _(type1, float, FLOAT)   \
      DEF_IF_DOUBLE(_(type1, double, DOUBLE)) \
      /* _(type1, int, FLOAT) */   \
      /* _(type1, long, DOUBLE) */ \
    } \
    break; \
  }



__kernel void operation_3_s(__global const void* a, __global const void* other, __global void* out, __global const void* alpha, const enum OpenCLOperationsPointwise3s op, const enum OpenCLPtrType typeTensor, const enum OpenCLPtrType typeAlpha) { \
  switch(typeTensor) {
    OPERATION_CASE_(bool, BOOL, typeAlpha, OPERATION_CASE)
    OPERATION_CASE_(char, CHAR, typeAlpha, OPERATION_CASE)
    OPERATION_CASE_(short, SHORT, typeAlpha, OPERATION_CASE)
    OPERATION_CASE_(int, INT, typeAlpha, OPERATION_CASE)
    OPERATION_CASE_(long, LONG, typeAlpha, OPERATION_CASE)
    OPERATION_CASE_(float, FLOAT, typeAlpha, OPERATION_CASE)
    DEF_IF_DOUBLE(OPERATION_CASE_(double, DOUBLE, typeAlpha, OPERATION_CASE))
    DEF_IF_NOT_DOUBLE(case DOUBLE: {break;})
    // OPERATION_CASE_(int, FLOAT, typeAlpha, OPERATION_CASE)
    // OPERATION_CASE_(long, DOUBLE, typeAlpha, OPERATION_CASE)
  }
}


#undef OPERATION
#undef OPERATION_CASE
#undef OPERATION_CASE_
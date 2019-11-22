#include <OpenCLKernelMacros.clh>
#include <OpenCLOperations.h>

#define CAST_CASE_(type1, name1, t2, _) \
  case name1: { \
    switch(t2) { \
      _(type1, bool, BOOL) \
      _(type1, char, CHAR) \
      _(type1, short, SHORT) \
      _(type1, int, INT) \
      _(type1, long, LONG) \
      _(type1, float, FLOAT) \
      DEF_IF_DOUBLE(_, type1, double, DOUBLE) \
      /* _(type1, int, FLOAT) */ \
      /* _(type1, long, DOUBLE) */ \
    } \
    break; \
  }

#define CAST_CASE(type1, type2, name2) \
  case name2: { \
    ((__global type2*)b)[get_global_id(0)] = ((__global type1*)a)[get_global_id(0)]; \
    break; \
  }

__kernel void cast(__global const void *a, __global void *b, const enum OpenCLPtrType ta, const enum OpenCLPtrType tb) {
  switch(ta) {
    CAST_CASE_(bool, BOOL, tb, CAST_CASE)
    CAST_CASE_(char, CHAR, tb, CAST_CASE)
    CAST_CASE_(short, SHORT, tb, CAST_CASE)
    CAST_CASE_(int, INT, tb, CAST_CASE)
    CAST_CASE_(long, LONG, tb, CAST_CASE)
    CAST_CASE_(float, FLOAT, tb, CAST_CASE)
    DEF_IF_DOUBLE(CAST_CASE_, double, DOUBLE, tb, CAST_CASE)
    // CAST_CASE_(int, FLOAT, tb, CAST_CASE)
    // CAST_CASE_(long, DOUBLE, tb, CAST_CASE)
  }
}

#define CAST_CASE_S(type1, type2, name2) \
  case name2: { \
    ((__global type2*)b)[get_global_id(0)] = *((__global const type1*) a); \
    break; \
  }

// Scalar version
__kernel void cast_s(__global const void *a, __global void *b, const enum OpenCLPtrType ta, const enum OpenCLPtrType tb) {
  switch(ta) {
    CAST_CASE_(bool, BOOL, tb, CAST_CASE_S)
    CAST_CASE_(char, CHAR, tb, CAST_CASE_S)
    CAST_CASE_(short, SHORT, tb, CAST_CASE_S)
    CAST_CASE_(int, INT, tb, CAST_CASE_S)
    CAST_CASE_(long, LONG, tb, CAST_CASE_S)
    CAST_CASE_(float, FLOAT, tb, CAST_CASE_S)
    DEF_IF_DOUBLE(CAST_CASE_, double, DOUBLE, tb, CAST_CASE_S)
    // CAST_CASE_(int, FLOAT, tb, CAST_CASE_S)
    // CAST_CASE_(long, DOUBLE, tb, CAST_CASE_S)
  }
}

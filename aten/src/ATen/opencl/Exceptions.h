#pragma once

#include <ATen/Context.h>
#include <c10/util/Exception.h>
#include <c10/opencl/OpenCLException.h>

// See Note [CHECK macro]
#define AT_OPENCL_CHECK(EXPR, ...) C10_OPENCL_CHECK(EXPR, ##__VA_ARGS__)

/* TODO If we need to check for the OpenCL drivers, implement it here */

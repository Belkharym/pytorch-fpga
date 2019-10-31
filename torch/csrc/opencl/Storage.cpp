#define __STDC_FORMAT_MACROS

#include <torch/csrc/python_headers.h>
#include <structmember.h>

// See Note [TH abstraction violation]
//    - Used to get at allocator from storage
#include <TH/THTensor.hpp>
#include <torch/csrc/opencl/THOP.h>

#include <torch/csrc/opencl/override_macros.h>
#include <torch/csrc/copy_utils.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>

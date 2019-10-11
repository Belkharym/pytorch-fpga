#pragma once

#include <c10/core/Allocator.h>

namespace at { namespace opencl {

CAFFE2_API at::Allocator* getPinnedMemoryAllocator();

}} // namespace at::opencl

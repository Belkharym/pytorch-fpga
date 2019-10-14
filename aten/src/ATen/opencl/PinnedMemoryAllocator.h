#pragma once

#include <c10/core/Allocator.h>
#include <c10/opencl/OpenCLStream.h>

namespace at { namespace opencl {

//
// A caching allocator for OpenCL host allocations (pinned memory).
//
// This provides a drop-in replacement for OpenCLHostAllocator, which re-uses
// freed pinned (page-locked) memory allocations. This avoids device
// synchronizations due to cudaFreeHost calls.
//
// To ensure correct behavior, OpenCLCachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a cudaMemcpyAsync
// call between host and device. We implement this for storages and tensors in
// copy_from_cpu_async_ and copy_to_cpu_async_.
//
// Note that this allocator does not split larger allocations into smaller
// blocks, unlike the caching device allocator.
//
CAFFE2_API at::Allocator* getPinnedMemoryAllocator();

// Records an event in the specified stream. The allocation 'ptr' will not be
// re-used until the event has occurred.
CAFFE2_API cl_int OpenCLCachingHostAllocator_recordEvent(void *ptr, at::opencl::OpenCLStream stream);

// Releases cached pinned memory allocations via cudaHostFree
CAFFE2_API void OpenCLCachingHostAllocator_emptyCache(void);

CAFFE2_API bool OpenCLCachingHostAllocator_isPinnedPtr(void *ptr);


}} // namespace at::opencl

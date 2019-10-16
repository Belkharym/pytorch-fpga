#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC

#include <c10/opencl/OpenCLStream.h>
#include <c10/core/Allocator.h>
#include <c10/opencl/OpenCLMacros.h>
#include <c10/util/Registry.h>

#include <mutex>

namespace c10 {

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class C10_OPENCL_API FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() {};
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeOpenCLMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeOpenCLMemoryCallbacksRegistry, name, __VA_ARGS__);

namespace opencl {

// TODO: Turn this into an honest to goodness class. I briefly attempted to do
// this, but it was a bit irritating to figure out how to also correctly
// apply pimpl pattern so I didn't have to leak any internal implementation
// details in the header (OpenCLCachingAllocator could be made a pimpl, but
// you also need to appropriately define a class which is a subclass
// of Allocator. Not impossible, but required a bit more surgery than
// I wanted to do at the time.)

namespace OpenCLCachingAllocator {

C10_OPENCL_API void* raw_alloc(size_t nbytes);
C10_OPENCL_API void raw_delete(void* ptr);

C10_OPENCL_API Allocator* get();
C10_OPENCL_API void emptyCache();
C10_OPENCL_API void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock);
C10_OPENCL_API cl::Buffer* getBufferFromPtr(void *ptr);
C10_OPENCL_API void* getBaseAllocation(void *ptr, size_t *size);
C10_OPENCL_API void recordStream(void *ptr, OpenCLStream stream);
C10_OPENCL_API uint64_t currentMemoryAllocated(int device);
C10_OPENCL_API uint64_t maxMemoryAllocated(int device);
C10_OPENCL_API void     resetMaxMemoryAllocated(int device);
C10_OPENCL_API uint64_t currentMemoryCached(int device);
C10_OPENCL_API uint64_t maxMemoryCached(int device);
C10_OPENCL_API void     resetMaxMemoryCached(int device);

C10_OPENCL_API std::mutex* getFreeMutex();

C10_OPENCL_API std::shared_ptr<void> getIpcDevPtr(std::string handle);

} // namespace OpenCLCachingAllocator

}} // namespace c10::opencl

#endif

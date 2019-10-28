#include <ATen/opencl/PinnedMemoryAllocator.h>
#include <ATen/opencl/OpenCLContext.h>
#include <ATen/Context.h>
#include <ATen/Config.h>
#include <ATen/opencl/detail/OpenCLHooks.h>

#include <stdexcept>

#ifdef _WIN32
#define ALIGNED_MALLOC(size, alignment) ::_aligned_malloc(size, alignment)
#else
#define ALIGNED_MALLOC(size, alignment) ::aligned_alloc(alignment, size)
#endif // _WIN32

namespace at { namespace opencl {

namespace {

struct BlockSize
{
  size_t  size; // allocation size
  void*   ptr;  // host memory pointer
  cl::Buffer buf; // buffer associated to host pointer

  BlockSize(size_t size, cl::Buffer buf = {}, void* ptr=NULL) : size(size), ptr(ptr), buf(buf) {}
};

struct Block : public BlockSize
{
  bool  allocated;    // true if the block is currently allocated
  int   event_count;  // number of outstanding cuda events
  std::unordered_set<at::opencl::OpenCLStream> streams;

  Block(size_t size, cl::Buffer buf, void* ptr, bool allocated) :
      BlockSize(size, buf, ptr), allocated(allocated), event_count(0), streams() {}
};

static bool BlockComparator(const BlockSize& a, const BlockSize& b)
{
  // sort by size, break ties with pointer
  if (a.size != b.size) {
    return a.size < b.size;
  }
  return (uintptr_t)a.ptr < (uintptr_t)b.ptr;
}

struct HostAllocator
{
  typedef bool (*Comparison)(const BlockSize&, const BlockSize&);

  // lock around all operations
  std::mutex mutex;

  // blocks by pointer
  std::unordered_map<void*, Block> blocks;

  // pointers that are ready to be allocated (event_count=0)
  std::set<BlockSize, Comparison> available;

  // outstanding opencl events
  std::deque<std::pair<cl::Event, void*>> opencl_events;

  HostAllocator() : available(BlockComparator) {}

  cl_int malloc(void** ptr, size_t size)
  {
    std::lock_guard<std::mutex> lock(mutex);

    // process outstanding opencl events which may have occurred
    cl_int err = processEvents();
    if (err != CL_SUCCESS) {
      return err;
    }

    // search for the smallest block which can hold this allocation
    BlockSize search_key(size);
    auto it = available.lower_bound(search_key);
    if (it != available.end()) {
      Block& block = blocks.at(it->ptr);
      TORCH_INTERNAL_ASSERT(!block.allocated && block.event_count == 0);
      block.allocated = true;
      *ptr = block.ptr;
      available.erase(it);
      return CL_SUCCESS;
    }

    // Pinned memory pointers allocated by any device can be directly used by any
    // other device, regardless of the current device at the time of allocation,
    // since we assume unified addressing.
    // So we grab any existing primary context, if available.
    // See pytorch/pytorch#21081.
    at::OptionalDeviceGuard device_guard;
    // TODO OpenCL Currently support 1 context. In the future, it might be possible
    // that we use more contexts (a context for GPUs, an other one for FPGAs, ...).
    // We might need to run a code similar to CUDA here.
    //auto primary_ctx_device_index = at::detail::getCUDAHooks().getDevceIndexWithPrimaryContext();
    //if (primary_ctx_device_index.has_value()) {
    //  device_guard.reset_device(at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
    //}
    device_guard.reset_device(at::Device(at::DeviceType::OPENCL, at::opencl::current_device()));

    // note that cudaHostAlloc may not touch pointer if size is 0
    *ptr = 0;

    // We allign the memory to 16 * MAX_SIZE_COMPONENT to meet the requirement of OpenCL alignement rull.
    // See https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/dataTypes.html
    *ptr = ALIGNED_MALLOC(size, alignof(max_align_t) * 16);
    TORCH_INTERNAL_ASSERT(*ptr, "Could not allocate memory for Host pointer.");

    cl::Buffer buffer{at::opencl::opencl_context(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, size, *ptr, &err};
    if (err != CL_SUCCESS) {
      return err;
    }

    blocks.insert({*ptr, Block(size, buffer, *ptr, true)});
    return CL_SUCCESS;
  }

  cl_int free(void* ptr)
  {
    std::lock_guard<std::mutex> lock(mutex);

    if (!ptr) {
      return CL_SUCCESS;
    }

    // process outstanding opencl events which may have occurred
    cl_int err = processEvents();
    if (err != CL_SUCCESS) {
      return err;
    }

    auto it = blocks.find(ptr);
    TORCH_INTERNAL_ASSERT(it != blocks.end());

    Block& block = it->second;
    TORCH_INTERNAL_ASSERT(block.allocated);

    // free (on valid memory) shouldn't fail, so mark unallocated before
    // we process the streams.
    block.allocated = false;

    // insert OpenCL events for each stream on which this block was used. This
    err = insertEvents(block);
    if (err != CL_SUCCESS) {
      return err;
    }

    if (block.event_count == 0) {
      // the block can be re-used if there are no outstanding opencl events
      available.insert(block);
    }
    return CL_SUCCESS;
  }

  cl_int recordEvent(void* ptr, at::opencl::OpenCLStream stream)
  {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = blocks.find(ptr);
    if (it == blocks.end()) {
      // ignore events for untracked pointers
      return CL_SUCCESS;
    }

    Block& block = it->second;
    TORCH_INTERNAL_ASSERT(block.allocated);

    block.streams.insert(stream);
    return CL_SUCCESS;
  }

  cl_int processEvents()
  {
    // Process outstanding clEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!opencl_events.empty()) {
      auto& e = opencl_events.front();
      cl::Event event = e.first;

      // The status could not change if getInfo returns an error.
      cl_int status = CL_COMPLETE;
      cl_int err = event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status);
      if (status != CL_COMPLETE) {
        break;
      } else if (err != CL_SUCCESS) {
        return err;
      }
      // Event implicitly destroyed when poped from opencl_events

      Block& block = blocks.at(e.second);
      block.event_count--;
      if (block.event_count == 0 && !block.allocated) {
        available.insert(block);
      }
      opencl_events.pop_front();
    }
    return CL_SUCCESS;
  }

  void emptyCache()
  {
    std::lock_guard<std::mutex> lock(mutex);

    // remove events for freed blocks
    for (auto it = opencl_events.begin(); it != opencl_events.end(); ++it) {
      cl::Event event = it->first;
      Block& block = blocks.at(it->second);
      if (!block.allocated) {
        // events implicitly destroyed on clear of opencl_events
        block.event_count--;
      }
    }

    // all opencl_events have been processed
    opencl_events.clear();

    // clear list of available blocks
    available.clear();

    auto stream = at::opencl::getCurrentOpenCLStream();
    // free and erase non-allocated blocks
    for (auto it = blocks.begin(); it != blocks.end();) {
      Block& block = it->second;
      if (!block.allocated) {
        ::free(block.ptr);
        it = blocks.erase(it);
      } else {
        ++it;
      }
    }
  }

  cl_int insertEvents(Block& block)
  {
    cl_int err = CL_SUCCESS;

    int prev_device;
    prev_device = at::opencl::current_device();

    std::unordered_set<at::opencl::OpenCLStream> streams(std::move(block.streams));
    for (auto it = streams.begin(); it != streams.end(); ++it) {
      at::opencl::set_device(it->device_index());

      cl::Event event;
      err = it->stream()->enqueueMarkerWithWaitList(NULL, &event);
      if (err != CL_SUCCESS) break;

      block.event_count++;
      opencl_events.emplace_back(event, block.ptr);
    }

    at::opencl::set_device(prev_device);
    return err;
  }
};

}  // namespace

static HostAllocator allocator;

cl_int OpenCLCachingHostAllocator_recordEvent(void *ptr, at::opencl::OpenCLStream stream)
{
  return allocator.recordEvent(ptr, stream);
}

void OpenCLCachingHostAllocator_emptyCache()
{
  allocator.emptyCache();
}

bool OpenCLCachingHostAllocator_isPinnedPtr(void *ptr) {
  return allocator.blocks.find(ptr) != allocator.blocks.end();
}

static void OpenCLCachingHostDeleter(void* ptr) {
  allocator.free(ptr);
}

struct OpenCLCachingHostAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    TORCH_INTERNAL_ASSERT(size >= 0);
    void *ptr;
    C10_OPENCL_CHECK(allocator.malloc(&ptr, size));
    return {ptr, ptr, &OpenCLCachingHostDeleter, at::DeviceType::CPU};
  }
  at::DeleterFnPtr raw_deleter() const override {
    return &OpenCLCachingHostDeleter;
  }
};

static OpenCLCachingHostAllocator opencl_caching_host_allocator;
at::Allocator* getPinnedMemoryAllocator() {
  return &opencl_caching_host_allocator;
}

cl::Buffer* OpenCLCachingHostAllocator_getBuffer(void *ptr) {
    auto it = allocator.blocks.find(ptr);
    if (it == allocator.blocks.end()) {
      return nullptr;
    }
    return &it->second.buf;
}

}} // namespace at::opencl

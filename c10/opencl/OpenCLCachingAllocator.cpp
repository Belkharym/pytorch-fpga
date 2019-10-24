#include <c10/opencl/OpenCLCachingAllocator.h>

#include <c10/opencl/OpenCLGuard.h>
#include <c10/opencl/OpenCLException.h>
#include <c10/opencl/OpenCLFunctions.h>
#include <c10/util/UniqueVoidPtr.h>
#include <ATen/opencl/OpenCLContext.h>
#include <ATen/opencl/OpenCLEvent.h>

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#define ALIGNED_MALLOC(size, alignment) ::_aligned_malloc(size, alignment)
#else
#define ALIGNED_MALLOC(size, alignment) ::aligned_alloc(alignment, size)
#endif // _WIN32

namespace c10 {

C10_DEFINE_REGISTRY(FreeOpenCLMemoryCallbacksRegistry, FreeMemoryCallback);

namespace opencl {
namespace OpenCLCachingAllocator {

//
// Yet another caching allocator for OpenCL device allocations.
//
// - Allocations are associated with a context. Once freed, blocks can be
//   re-allocated on the same context, but not on any other context.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to clCreateBuffer.
// - If the clCreateBuffer fails, the allocator will free all cached blocks that
//   are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using clCreateBuffer.
//   To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//



namespace {

using stream_set = std::unordered_set<opencl::OpenCLStream>;

constexpr size_t kMinBlockSize = 512;       // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;      // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;    // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;   // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;     // round up large allocs to 2 MiB

struct DeviceStats {
  uint64_t   amount_allocated;      // total amount allocated in bytes
  uint64_t   max_amount_allocated;  // max total amount allocated in bytes
  uint64_t   amount_cached;         // total amount in cache in bytes
  uint64_t   max_amount_cached;     // max total amount in cache in bytes

  DeviceStats() :
      amount_allocated(0), max_amount_allocated(0),
      amount_cached(0), max_amount_cached(0) { }

  void increaseAllocated(size_t delta) {
    amount_allocated += delta;
    max_amount_allocated = std::max(max_amount_allocated, amount_allocated);
  }

  void decreaseAllocated(size_t delta) {
    amount_allocated -= delta;
  }

  void increaseCached(size_t delta) {
    amount_cached += delta;
    max_amount_cached = std::max(max_amount_cached, amount_cached);
  }

  void decreaseCached(size_t delta) {
    amount_cached -= delta;
  }
};

struct Block;
typedef bool (*Comparison)(const Block*, const Block*);
typedef std::set<Block*, Comparison> BlockPool;

struct Block {
  int           device;      // gpu
  CommandQueue_t  stream;      // allocation stream
  stream_set    stream_uses; // streams on which the block was used
  size_t        size;        // block size in bytes
  BlockPool*    pool;        // owning memory pool
  cl::Buffer*   buf;         // Buffer associated to ptr
  void*         ptr;         // memory address
  bool          allocated;   // in-use flag
  Block*        prev;        // prev block if split from a larger allocation
  Block*        next;        // next block if split from a larger allocation
  int           event_count; // number of outstanding OpenCL events

  Block(int device, CommandQueue_t stream, size_t size, BlockPool* pool, cl::Buffer* buf, void* ptr) :
    device(device), stream(stream), stream_uses(), size(size), pool(pool),
    buf(buf), ptr(ptr), allocated(0), prev(nullptr), next(nullptr), event_count(0) { }

  // constructor for search key
  Block(int device, CommandQueue_t stream, size_t size) :
    device(device), stream(stream), stream_uses(), size(size), pool(nullptr),
    ptr(nullptr), allocated(0), prev(nullptr), next(nullptr), event_count(0) { }
};

static bool BlockComparator(const Block* a, const Block* b)
{
  if (a->device != b->device) {
    return a->device < b->device;
  }
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

} // namespace

struct THOCachingAllocator
{
  // device statistics
  std::vector<DeviceStats> device_stats;

  // lock around all operations
  std::recursive_mutex mutex;

  // lock around calls to clReleaseMemObject
  std::mutex opencl_free_mutex;

  // cached blocks larger than 1 MB
  BlockPool large_blocks;

  // cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated blocks by device pointer
  std::unordered_map<void*, Block*> allocated_blocks;

  // outstanding opencl events
  std::deque<std::pair<cl::Event, Block*>> opencl_events;

  THOCachingAllocator() :
      large_blocks(BlockComparator),
      small_blocks(BlockComparator) {}

  DeviceStats &get_stats_for_device(int device) {
    AT_ASSERT(device >= 0);
    if ((size_t) device >= device_stats.size()) {
      device_stats.resize(device + 1);
    }
    return device_stats.at(device);
  }

  /** allocates a block which is safe to use from the provided stream */
  void malloc(void** devPtr, size_t size, CommandQueue_t stream)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    int device = c10::opencl::current_device();

    // process outstanding opencl Events
    process_events();

    size = round_size(size);

    DeviceStats &stats = get_stats_for_device(device);

    Block search_key(device, stream, size);
    auto& pool = get_pool(size);

    auto find_free_block = [&]()->Block*{
      auto it = pool.lower_bound(&search_key);
      if (it != pool.end() && (*it)->device == device &&
          (*it)->stream == stream) {
        Block* block = *it;
        pool.erase(it);
        return block;
      }
      return nullptr;
    };

    Block* block = find_free_block();
    if (block == nullptr) {
      bool freed_memory = false;
      for (const auto& name : FreeOpenCLMemoryCallbacksRegistry()->Keys()) {
        freed_memory |=
            FreeOpenCLMemoryCallbacksRegistry()->Create(name)->Execute();
      }
      if (freed_memory) {
        block = find_free_block();
      }
    }
    if (block == nullptr) {
      void* ptr;
      cl::Buffer buf;
      size_t alloc_size = get_allocation_size(size);
      cl_int err = opencl_malloc_retry(device, &buf, &ptr, alloc_size);
      if (err != CL_SUCCESS) {
        if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
          size_t device_free = 0;
          size_t device_total;
          auto device_prop = at::opencl::getDeviceProperties(device);
          device_total = device_prop->globalMemSize;
          const auto& stats = get_stats_for_device(device);
          device_free = device_total - stats.amount_allocated;

          // "total capacity": total global memory on Device
          // "already allocated": memory allocated by the program using the
          //                      caching allocator
          // "free": free memory as reported by the OpenCL API
          // "cached": memory held by the allocator but not used by the program
          //
          // The "allocated" amount  does not include memory allocated outside
          // of the caching allocator, such as memory allocated by other programs
          // or memory held by the driver.
          //
          // The sum of "allocated" + "free" + "cached" may be less than the
          // total capacity due to memory held by the driver and usage by other
          // programs.
          //
          // Note that at this point opencl_malloc_retry has already returned all
          // possible "cached" memory to the driver. The only remaining "cached"
          // memory is split from a larger block that is partially in-use.
          AT_ERROR(
            "OpenCL out of memory. Tried to allocate ", format_size(alloc_size),
            " (Device ", device_prop->name, "; ",
            format_size(device_total), " total capacity; ",
            format_size(stats.amount_allocated), " already allocated; ",
            format_size(device_free), " free; ",
            format_size(stats.amount_cached - stats.amount_allocated), " cached)");
        } else {
          C10_OPENCL_CHECK(err);
        }
      }
      stats.increaseCached(alloc_size);
      block = new Block(device, stream, alloc_size, &pool, new cl::Buffer{std::move(buf)}, ptr);
    }

    Block* remaining = nullptr;
    AT_ASSERT(block);
    if (should_split(block, size)) {

      remaining = block;

      block = new Block(device, stream, size, &pool, block->buf, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      pool.insert(remaining);
    }

    block->allocated = true;
    allocated_blocks[block->ptr] = block;

    *devPtr = block->ptr;

    stats.increaseAllocated(block->size);
  }

  void free(void* ptr)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (!ptr) {
      return;
    }

    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      AT_ERROR("invalid device pointer: ", ptr);
    }

    Block* block = it->second;
    allocated_blocks.erase(it);
    block->allocated = false;

    get_stats_for_device(block->device).decreaseAllocated(block->size);
    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block);
    }
  }

  /** returns cached blocks to the system allocator */
  void emptyCache()
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    synchronize_and_free_events(nullopt);
    free_blocks(large_blocks, large_blocks.begin(), large_blocks.end());
    free_blocks(small_blocks, small_blocks.begin(), small_blocks.end());
  }

  void* getBaseAllocation(void* ptr, size_t* outSize)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    Block* block = find_allocated_block(ptr);
    if (!block) {
      AT_ERROR("invalid device pointer: ", ptr);
    }
    while (block->prev) {
      block = block->prev;
    }
    void *basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  // Accumulates sizes of all memory blocks for given device in given pool
  void cacheInfoAux(BlockPool& blocks, int dev_id, size_t* total, size_t* largest)
  {
    Block search_key(dev_id, 0, 0);
    auto it = blocks.lower_bound(&search_key);
    for (; it != blocks.end() && *it && (*it)->device == dev_id; ++it) {
      size_t blocksize = (*it)->size;
      *total += blocksize;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

  void cacheInfo(int dev_id, size_t* total, size_t* largest)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    cacheInfoAux(large_blocks, dev_id, total, largest);
    cacheInfoAux(small_blocks, dev_id, total, largest);
  }

  void recordStream(void* ptr, opencl::OpenCLStream stream)
  {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (ptr) {
      std::lock_guard<std::recursive_mutex> lock(mutex);
      Block* block = find_allocated_block(ptr);
      // block could be nullptr in some cases, e.g., tensor loaded from blob, or
      // shared from another process, or not pointing to a OpenCL tensor.
      if (block) {
        if (stream.stream() == block->stream) {
          // ignore uses on the allocation stream, since those don't require any
          // special synchronization
          return;
        }
        block->stream_uses.insert(stream);
      }
    }
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(Block* block)
  {
    AT_ASSERT(!block->allocated && block->event_count == 0);
    auto& pool = *block->pool;
    try_merge_blocks(block, block->prev, pool);
    try_merge_blocks(block, block->next, pool);
    pool.insert(block);
  }

  /** combine previously split blocks */
  void try_merge_blocks(Block* dst, Block* src, BlockPool& pool)
  {
    if (!src || src->allocated || src->event_count > 0) {
      return;
    }
    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    dst->size += src->size;
    pool.erase(src);
    delete src;
  }

  BlockPool& get_pool(size_t size) {
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  bool should_split(Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool == &small_blocks) {
      return remaining >= kMinBlockSize;
    } else if (block->pool == &large_blocks) {
      return remaining > kSmallSize;
    } else {
      AT_ERROR("should_split: invalid pool");
    }
  }

  size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

  size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  cl_int opencl_malloc_retry(int device, cl::Buffer* devBuf, void** devPtr, size_t size)
  {
    // Try clCreateBuffer. If clCreateBuffer fails, frees all non-split cached blocks
    // and retries.
    cl_int err;
    *devPtr = ALIGNED_MALLOC(size, alignof(max_align_t) * 16);
    if (*devPtr == nullptr) {
      C10_OPENCL_CHECK(false, "Cannot allocate ", size, " byte for OpenCL Buffer");
    }
    *devBuf = cl::Buffer{c10::opencl::opencl_context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, devPtr, &err};
    if (err != CL_SUCCESS) {
      free_cached_blocks(device);
      *devBuf = cl::Buffer{c10::opencl::opencl_context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, devPtr, &err};
      if (err != CL_SUCCESS) {
        ::free(*devPtr);
        return err;
      }
    }
    return CL_SUCCESS;
  }

  void free_cached_blocks(int device)
  {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(device);

    // Free all non-split cached blocks on device
    Block lower_bound(device, nullptr, 0);
    Block upper_bound(device + 1, nullptr, 0);

    free_blocks(
        large_blocks,
        large_blocks.lower_bound(&lower_bound),
        large_blocks.lower_bound(&upper_bound));
    free_blocks(
        small_blocks,
        small_blocks.lower_bound(&lower_bound),
        small_blocks.lower_bound(&upper_bound));
  }

  void free_blocks(BlockPool& blocks, BlockPool::iterator it, BlockPool::iterator end)
  {
    // Frees all non-split blocks between `it` and `end`
    std::lock_guard<std::mutex> lock(opencl_free_mutex);
    while (it != end) {
      Block* block = *it;
      if (!block->prev && !block->next) {
        delete block->buf;
        ::free((void*)block->ptr);
        get_stats_for_device(block->device).decreaseCached(block->size);
        auto cur = it;
        ++it;
        blocks.erase(cur);
        delete block;
      } else {
        ++it;
      }
    }
  }

  void synchronize_and_free_events(optional<int> device) {
    // Synchronize on outstanding events and then free associated blocks.
    // Limited to blocks on the given device if specified.

    auto remaining_events = decltype(opencl_events)();

    for (auto& e : opencl_events) {
      cl::Event event = e.first;
      Block* block = e.second;
      if (device.has_value() && block->device != *device) {
        remaining_events.push_back(e);
        continue;
      }

      C10_OPENCL_CHECK(event.wait());

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
    }

    std::swap(opencl_events, remaining_events);
  }

  Block* find_allocated_block(void *ptr) {
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    return it->second;
  }

  void insert_events(Block* block)
  {
    int prev_device = c10::opencl::current_device();

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto it = streams.begin(); it != streams.end(); ++it) {
      c10::opencl::set_device(it->device_index());

      cl::Event event;
      C10_OPENCL_CHECK(it->stream()->enqueueMarkerWithWaitList(NULL, &event));

      block->event_count++;
      opencl_events.emplace_back(event, block);
    }

    c10::opencl::set_device(prev_device);
  }

  void process_events()
  {
    // Process outstanding opencl Events. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!opencl_events.empty()) {
      auto& e = opencl_events.front();
      cl::Event event = e.first;
      Block* block = e.second;

      cl_int err;
      cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>(&err);
      if (err == CL_SUCCESS && status != CL_COMPLETE) {
        // ignore if not ready
        break;
      } else if (err != CL_SUCCESS) {
        C10_OPENCL_CHECK(err);
      }

      // Event implicitly destroyed when poped from opencl_events

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
      opencl_events.pop_front();
    }
  }
};

THOCachingAllocator caching_allocator;

static void OpenCLCachingDeleter(void* ptr) {
  caching_allocator.free(ptr);
}

// NB: I decided not to fold this into THCCachingAllocator, because the latter
// has a lot more methods and it wasn't altogether clear that they should
// actually be publically exposed
struct OpenCLCachingAllocator : public Allocator {
  DataPtr allocate(size_t size) const override {
    int device = c10::opencl::current_device();
    void* r = nullptr;
    if (size != 0) {
      caching_allocator.malloc(&r, size, opencl::getCurrentOpenCLStream(device));
    }
    return {r, r, &OpenCLCachingDeleter, Device(DeviceType::OPENCL, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    return &OpenCLCachingDeleter;
  }
};

OpenCLCachingAllocator device_allocator;

Allocator* get(void)
{
  return &device_allocator;
}

void emptyCache(void) {
  caching_allocator.emptyCache();
}

void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
  caching_allocator.cacheInfo(dev_id, cachedAndFree, largestBlock);
}

cl::Buffer* getBufferFromPtr(void *ptr) {
  Block* block = caching_allocator.find_allocated_block(ptr);
  if (block != nullptr) {
    return block->buf;
  }
  return nullptr;
}

void* getBaseAllocation(void *ptr, size_t *size)
{
  return caching_allocator.getBaseAllocation(ptr, size);
}

void recordStream(void *ptr, opencl::OpenCLStream stream)
{
  caching_allocator.recordStream(ptr, stream);
}

std::mutex* getFreeMutex()
{
  return &caching_allocator.opencl_free_mutex;
}

static inline void assertValidDevice(int device) {
  int device_num = device_count();
  AT_ASSERTM(0 <= device && device < device_num, "Invalid device argument.");
}

uint64_t currentMemoryAllocated(int device)
{
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).amount_allocated;
}

uint64_t maxMemoryAllocated(int device) {
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).max_amount_allocated;
}

void resetMaxMemoryAllocated(int device) {
  assertValidDevice(device);
  DeviceStats& stats = caching_allocator.get_stats_for_device(device);
  stats.max_amount_allocated = stats.amount_allocated;
}

uint64_t currentMemoryCached(int device)
{
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).amount_cached;
}

uint64_t maxMemoryCached(int device) {
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).max_amount_cached;
}

void resetMaxMemoryCached(int device) {
  assertValidDevice(device);
  DeviceStats& stats = caching_allocator.get_stats_for_device(device);
  stats.max_amount_cached = stats.amount_cached;
}

void* raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device = opencl::current_device();
  void* r = nullptr;
  caching_allocator.malloc(&r, nbytes, opencl::getCurrentOpenCLStream(device));
  return r;
}

void raw_delete(void* ptr) {
  caching_allocator.free(ptr);
}

} // namespace OpenCLCachingAllocator

}} // namespace c10::opencl

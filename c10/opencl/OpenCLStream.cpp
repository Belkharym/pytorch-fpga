#include "OpenCLStream.h"
#include <c10/opencl/OpenCLGuard.h>
#include <c10/opencl/OpenCLFunctions.h>
#include <c10/util/Exception.h>

#include <mutex>
#include <atomic>

namespace c10 {
namespace opencl {
namespace {

/* TODO Reimplement LeakyStreamInternals to work with OpenCL */

// Internal implementation that leaks the stream. It's not intended to be used
// outside of this file.
struct LeakyStreamInternals {
  LeakyStreamInternals() = default;
  C10_DISABLE_COPY_AND_ASSIGN(LeakyStreamInternals);

  ~LeakyStreamInternals() {
    // NB: this code is invoked only in the destruction of global variables
    // (since we never shrink the corresponding vectors). At this point the CUDA
    // runtime might be already destroyed and invoking cudaStreamDestroy leads
    // to a crash. It's likely an issue in CUDA, but to be safe - let's just
    // "forget" the destruction.

    if (stream) delete stream;
  }

  c10::DeviceIndex device_index = -1;
  int32_t stream_id = -1;
  CommandQueue_t stream = nullptr;
};
// Default streams
static std::once_flag init_flag;
static LeakyStreamInternals default_streams[C10_COMPILE_TIME_MAX_OPENCL_DEVICES];

// Global stream state and constants
static DeviceIndex num_devices = -1;
static constexpr int kStreamsPerPoolBits = 5;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;

static std::once_flag device_flags[C10_COMPILE_TIME_MAX_OPENCL_DEVICES];
static std::atomic<uint32_t> pool_counters[C10_COMPILE_TIME_MAX_OPENCL_DEVICES];
static std::array<LeakyStreamInternals, kStreamsPerPool>
    pool_streams[C10_COMPILE_TIME_MAX_OPENCL_DEVICES];

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 26 bits -- -- 1 bits --  -- 5 bits -----
// zeros         StreamIdType  stream id index
//
// Where StreamIdType:
//  0 = default stream
//  1 = pool stream
//
// This is not really for efficiency; it's just easier to write the code
// to extract the index if we do this with bitmasks :)
//
// We are obligated to treat the stream ID 0 as the default stream, per the
// invariant specified in c10::Stream.  However, all other numbers are entirely
// an internal implementation detail, we reserve the right to renumber streams
// however we like.
//
// Note that it is really important that the MSB is zero; StreamId is a
// *signed* integer, and unsigned to signed conversion outside of the
// bounds of signed integer representation is undefined behavior.  You
// could work around this with something like
// https://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior
// but it seems a bit overkill for this.
enum class StreamIdType : uint8_t {
  DEFAULT = 0x0,
  POOL = 0x1,
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  switch (s) {
    case StreamIdType::DEFAULT:
      stream << "DEFAULT";
      break;
    case StreamIdType::POOL:
      stream << "POOL";
      break;
    default:
      stream << static_cast<uint8_t>(s);
      break;
  }
  return stream;
}

// StreamId is 32-bit, so we can just rely on regular promotion rules.
// We rely on streamIdIndex and streamIdType being non-negative;
// see Note [Hazard when concatenating signed integers]

static inline StreamIdType streamIdType(StreamId s) {
  return static_cast<StreamIdType>(s >> kStreamsPerPoolBits);
}

static inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(s & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  return (static_cast<StreamId>(st) << kStreamsPerPoolBits) |
      static_cast<StreamId>(si);
}

template <typename T, typename A>
static bool pointer_within(const T* ptr, const A& arr) {
  return std::greater_equal<const T*>()(ptr, arr.data()) &&
      std::less<const T*>()(ptr, arr.data() + arr.size());
}

static StreamId OpenCLStream_getStreamId(const LeakyStreamInternals* ptr) {
  // Hypothetically, we could store the stream ID in the stream.  But that
  // introduces a degree of freedom which could lead to bugs (where we
  // misnumber streams in the pool, or overwrite the number).  Better
  // to just compute it based on the metric that actually matters,
  // which is how we map IDs back into the vectors.

  DeviceIndex device_index = ptr->device_index;

  // Check if it's the default stream
  if (ptr == &default_streams[device_index]) {
    return makeStreamId(StreamIdType::DEFAULT, 0);
  }

  // Check if it's a pool stream
  // NB: Because ptr may not necessarily lie within the array, we must use
  // std::less and similar templates to avoid UB that arises when
  // doing an operator< comparison.cuda
  if (pointer_within<LeakyStreamInternals>(
          ptr, pool_streams[device_index])) {
    return makeStreamId(
        StreamIdType::POOL, ptr - pool_streams[device_index].data());
  }

  AT_ASSERTM(
      0,
      "Could not compute stream ID for ",
      ptr,
      " on device ",
      device_index,
      " (something has gone horribly wrong!)");
}

// Thread-local current streams
static thread_local LeakyStreamInternals** current_streams = nullptr;

// Populates global values and creates a default stream for each device.
// Note: the default stream on each device is signified by a nullptr,
// and so is not created as usual.
// In particular, we don't need to switch devices when creating the
// streams.
// Warning: this function must only be called once!
static void initGlobalStreamState() {
  num_devices = c10::opencl::device_count();
  // Check if the number of devices matches the expected compile-time max number
  // of devices.
  AT_ASSERTM(
      num_devices <= C10_COMPILE_TIME_MAX_OPENCL_DEVICES,
      "Number of OpenCL devices on the machine is larger than the compiled "
      "max number of gpus expected (",
      C10_COMPILE_TIME_MAX_OPENCL_DEVICES,
      "). Increase that and recompile.");

  // Initializes default streams
  for (auto i = decltype(num_devices){0}; i < num_devices; ++i) {
    OpenCLGuard guard{i};
    default_streams[i].device_index = i;
    default_streams[i].stream_id = 0;
    default_streams[i].stream = new cl::CommandQueue(opencl_context(), opencl_device(i));
    pool_counters[i] = 0;
  }
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
static void initDeviceStreamState(DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  c10::opencl::OpenCLGuard device_guard{device_index};

  for (auto i = decltype(kStreamsPerPool){0}; i < kStreamsPerPool; ++i) {
    auto& stream = pool_streams[device_index][i];

    stream.device_index = device_index;

    stream.stream = new cl::CommandQueue(opencl_context(), opencl_device(device_index));
  }
}

// Init front-end to ensure initialization only occurs once
static void initOpenCLStreamsOnce() {
  // Inits default streams (once, globally)
  std::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  current_streams =
      (LeakyStreamInternals**)malloc(num_devices * sizeof(LeakyStreamInternals*));
  for (auto i = decltype(num_devices){0}; i < num_devices; ++i) {
    current_streams[i] = &default_streams[i];
  }
}

// Helper to verify the OpenCL device index is valid
static inline void check_device(DeviceIndex device_index) {
  AT_ASSERT(device_index >= 0 && device_index < num_devices);
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in OpenCLStream.h)
static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

// See Note [StreamId assignment]
LeakyStreamInternals* OpenCLStream_internals(OpenCLStream s) {
  c10::DeviceIndex device_index = s.device_index();
  StreamIdType st = streamIdType(s.unwrap().id());
  size_t si = streamIdIndex(s.unwrap().id());
  switch (st) {
    case StreamIdType::DEFAULT:
      AT_ASSERTM(
          si == 0,
          "Unrecognized stream ",
          s.unwrap(),
          " (I think this should be the default stream, but I got a non-zero index ",
          si,
          ").",
          " Did you manufacture the StreamId yourself?  Don't do that; use the",
          " official API like c10::cuda::getStreamFromPool() to get a new stream.");
      return &default_streams[device_index];
    case StreamIdType::POOL:
      return &pool_streams[device_index][si];
    default:
      AT_ASSERTM(
          0,
          "Unrecognized stream ",
          s.unwrap(),
          " (I didn't recognize the stream type, ",
          st,
          ")");
  }
}

OpenCLStream OpenCLStream_fromInternals(const LeakyStreamInternals* ptr) {
  return OpenCLStream(
      OpenCLStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::OPENCL, ptr->device_index),
          OpenCLStream_getStreamId(ptr)));
}

} // namespace <unnamed>

CommandQueue_t OpenCLStream::stream() const {
  auto ptr = OpenCLStream_internals(*this);
  AT_ASSERT(ptr);
  return ptr->stream;
}

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
OpenCLStream getStreamFromPool(DeviceIndex device_index) {
  initOpenCLStreamsOnce();
  if (device_index == -1)
    device_index = c10::opencl::current_device();
  check_device(device_index);

  // Initializes the stream pools (once)
  std::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);

  const auto idx = get_idx(pool_counters[device_index]);
  return OpenCLStream_fromInternals(&pool_streams[device_index][idx]);
}

OpenCLStream getDefaultOpenCLStream(DeviceIndex device_index) {
  initOpenCLStreamsOnce();
  if (device_index == -1) {
    device_index = c10::opencl::current_device();
  }
  check_device(device_index);
  return OpenCLStream_fromInternals(&default_streams[device_index]);
}
OpenCLStream getCurrentOpenCLStream(DeviceIndex device_index) {
  initOpenCLStreamsOnce();
  if (device_index == -1) {
    device_index = c10::opencl::current_device();
  }
  check_device(device_index);
  return OpenCLStream_fromInternals(current_streams[device_index]);
}

void setCurrentOpenCLStream(OpenCLStream stream) {
  initOpenCLStreamsOnce();
  auto ptr = OpenCLStream_internals(stream);
  AT_ASSERT(ptr);
  current_streams[ptr->device_index] = ptr;
}

cl_int openclSynchronize() {
  cl_int err = CL_SUCCESS;
  for (size_t i = 0; i < num_devices; ++i) {
    err = current_streams[i]->stream->flush();
    if (err != CL_SUCCESS) return err;
    err = current_streams[i]->stream->finish();
    if (err != CL_SUCCESS) return err;
  }
  return err;
}

std::ostream& operator<<(std::ostream& stream, const OpenCLStream& s) {
  return stream << s.unwrap();
}

} // namespace opencl
} // namespace c10

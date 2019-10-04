#ifndef CAFFE2_OPENCL_CONTEXT_H_
#define CAFFE2_OPENCL_CONTEXT_H_

#include "caffe2/core/context.h"
#include "caffe2/core/context_base.h"

#include <c10/core/Device.h>
#include <c10/opencl/OpenCLMacros.h>
#include <c10/opencl/OpenCLStream.h>

#define CL_HPP_ENABLE_EXCEPTIONS 1
#define CL_HPP_CL_1_2_DEFAULT_BUILD 1
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
//#include "libopencl.h"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#define OPENCL_CHECK(expr) (void)expr

namespace caffe2 {

namespace opencl {

/**
 * A struct to host thread-local opencl objects.
 *
 * In Caffe2, each thread has its own non-default opencl stream. This is achieved by
 * having the ThreadLocalOpenCLObjects wrapper that takes care of allocating
 * and deallocating these objects at the thread scope. This class is solely
 * used inside OpenCLContext and should not be used externally.
 *
 * This class manages the mapping from logical stream ID (int stream_id
 * passed around in Caffe2) and OpenCLStream objects.  We intend to eventually
 * deprecate the logical stream ID interface, but not for now.
 */
class CAFFE2_API ThreadLocalOpenCLObjects {
 friend class OpenCLContext;

 private:
    ThreadLocalOpenCLObjects() {
        for (DeviceIndex i = 0; i < C10_COMPILE_TIME_MAX_OPENCL_DEVICES; ++i) {
            opencl_streams_[i] = vector<c10::opencl::OpenCLStream>();
        }
    }

    // Record current stream id for the current thread.
    // This is the new API we're trying to migrate use cases to and get rid of
    // explicit stream id passing. For now it's invoked in
    // OpenCLContext::SwitchToDevice
    void SetCurrentStreamId(DeviceIndex device, StreamId stream_id) {
        // TODO: use current device id from thread local instead of passing device in
        c10::opencl::setCurrentOpenCLStream(GetOpenCLStream(device, stream_id));
    }

    // Retrieves the CUDAStream corresponding to a logical stream ID, ensuring
    // that it exists in opencl_streams_ if it has not been allocated yet.
    c10::opencl::OpenCLStream GetOpenCLStream(DeviceIndex device, StreamId stream_id) {
        vector<c10::opencl::OpenCLStream>& gpu_streams = opencl_streams_[device];
        while (gpu_streams.size() <= static_cast<size_t>(stream_id)) {
            // NB: This streams are not guaranteed to be unique; we'll
            // wrap around once we run out of streams in the pool.
            gpu_streams.emplace_back(c10::opencl::getStreamFromPool(device));
        }
        return gpu_streams[stream_id];
    }

    // Uses the logical stream id from the thread local to pick the stream
    // We're going to migrate all usages to this case API instead of passing the
    // stream id directly
    c10::opencl::CommandQueue_t GetStream(DeviceIndex device) {
        return c10::opencl::getCurrentOpenCLStream(device).stream();
    }

    ~ThreadLocalOpenCLObjects() noexcept {}

    // WARNING: mapping from logical stream ID to c10::opencl::OpenCLStream
    // is NOT bijective; multiple logical stream IDs may map to the
    // same underlying stream ID.
    vector<c10::opencl::OpenCLStream> opencl_streams_[C10_COMPILE_TIME_MAX_OPENCL_DEVICES];
};

class OpenCLContext final : public at::BaseContext {
public:
    explicit OpenCLContext();
    explicit OpenCLContext(const caffe2::DeviceOption& option) {
        DCHECK_EQ(option.device_type(), caffe2::DeviceTypeProto::PROTO_OPENCL);
        OpenCLContext();
    }
    explicit OpenCLContext(const Device& device) : OpenCLContext(DeviceToOption(device)) {}
    ~OpenCLContext() override {
        // OpenCLContext is used in 2 cases now:
        // - long-lived instance inside OperatorBase in which case what happens in
        //   destructor doesn't really matter
        // - short-lived on-the-fly instances that are utilized as OpenCLGuard - in
        //   this case there's only one stream id (passed to SwitchToDevice) and
        //   it's preferrable to synchronize in the destructor
        FinishDeviceComputation();
    }

    virtual Device device() const;

    /* Sorry for the naming, will get rid of this in future diff */
    virtual DeviceType device_type() const;

    virtual void SwitchToDevice(int stream_id);

    using BaseContext::SwitchToDevice;

    inline static at::DataPtr New(size_t nbytes) {
        return GetAllocator(OPENCL)->allocate(nbytes);
    }

    virtual void WaitEvent(const caffe2::Event& ev);

    virtual void Record(caffe2::Event* ev, const char* err_msg = NULL)
        const;

    virtual void FinishDeviceComputation();

    // shared mutex to lock out alloc / free during NCCL launches
    static std::mutex& mutex();

    // This used to be arbitrary cross-device copy, but it turns out everyone
    // did direct CPU-X copy, so we just make three functions for it (to avoid
    // double dispatch).  This will get obsoleted by C10. where copies
    // will be proper operators (and get to rely on multiple dispatch there.)
    virtual void CopyBytesSameDevice(
        size_t nbytes,
        const void* src,
        void* dst);

    virtual void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst);

    virtual void CopyBytesToCPU(size_t nbytes, const void* src, void* dst);

    static void CopyBytesAsync(
        size_t nbytes,
        const void* src,
        Device src_device,
        void* dst,
        Device dst_device);
    static void CopyBytesSync(
        size_t nbytes,
        const void* src,
        Device src_device,
        void* dst,
        Device dst_device);

    // By default CPU operators don't have async device parts
    static bool HasAsyncPartDefault() {
        return true;
    }

    static bool SupportsAsyncScheduling() {
        return true;
    }

    static bool IsStreamFree(
        const DeviceOption& /* option */,
        int /* stream_id */) {
        return true;
    }

    static constexpr DeviceType GetDeviceType() {
        return OPENCL;
    }

};

} // namespace opencl
} // namespace caffe2

#endif /* CAFFE2_OPENCL_CONTEXT_H_ */

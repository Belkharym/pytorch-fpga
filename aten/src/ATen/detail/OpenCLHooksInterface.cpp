#include <ATen/detail/OpenCLHooksInterface.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

// See getCUDAHooks for some more commentary
const OpenCLHooksInterface& getOpenCLHooks() {
  static std::unique_ptr<OpenCLHooksInterface> opencl_hooks;
  static std::once_flag once;
  std::call_once(once, [] {
    opencl_hooks = OpenCLHooksRegistry()->Create("OpenCLHooks", OpenCLHooksArgs{});
    if (!opencl_hooks) {
      opencl_hooks =
          std::unique_ptr<OpenCLHooksInterface>(new OpenCLHooksInterface());
    }
  });
  return *opencl_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(OpenCLHooksRegistry, OpenCLHooksInterface, OpenCLHooksArgs)

} // namespace at

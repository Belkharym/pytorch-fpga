#pragma once

namespace torch {
namespace utils {

static inline bool opencl_enabled() {
#ifdef USE_OPENCL
  return true;
#else
  return false;
#endif
}

}
}

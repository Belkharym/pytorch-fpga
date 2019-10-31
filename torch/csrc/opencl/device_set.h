#pragma once

#include <bitset>

namespace torch {

static constexpr size_t MAX_OPENCL_DEVICES = 64;
using device_set = std::bitset<MAX_OPENCL_DEVICES>;

}

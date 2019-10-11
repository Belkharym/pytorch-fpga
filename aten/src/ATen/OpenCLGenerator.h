#pragma once

#include <ATen/core/Generator.h>

namespace at {

struct CAFFE2_API OpenCLGenerator : public Generator {
  // Constructors
  OpenCLGenerator(DeviceIndex device_index = -1);
  ~OpenCLGenerator() = default;

  // CUDAGenerator methods
  std::shared_ptr<OpenCLGenerator> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread();
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
  static DeviceType device_type();

private:
  OpenCLGenerator* clone_impl() const override;
  uint64_t seed_ = default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
};

namespace opencl {
namespace detail {

  CAFFE2_API OpenCLGenerator* getDefaultOpenCLGenerator(DeviceIndex device_index = -1);
  CAFFE2_API std::shared_ptr<OpenCLGenerator> createOpenCLGenerator(DeviceIndex device_index = -1);

} // namespace detail
} // namespace opencl
} // namespace at


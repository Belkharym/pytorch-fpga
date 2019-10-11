#include <ATen/OpenCLGenerator.h>
#include <c10/opencl/OpenCLFunctions.h>

namespace at {

namespace opencl { namespace detail {

// Ensures we only call cudaGetDeviceCount only once.
static std::once_flag num_device_init_flag;

// Total number of gpus in the system.
static int64_t num_devices;

// Ensures default_gens_opencl is initialized once.
static std::deque<std::once_flag> opencl_gens_init_flag;

// Default, global OpenCL generators, one per Device.
static std::vector<std::shared_ptr<OpenCLGenerator>> default_gens_opencl;

/* 
* Populates the global variables related to OpenCL generators
* Warning: this function must only be called once!
*/
static void initOpenCLGenVector(){
  num_devices = c10::opencl::device_count();
  opencl_gens_init_flag.resize(num_devices);
  default_gens_opencl.resize(num_devices);
}

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultOpenCLGenerator gets the default generator for a particular
 * opencl device.
 */
OpenCLGenerator* getDefaultOpenCLGenerator(DeviceIndex device_index) {
  std::call_once(num_device_init_flag, initOpenCLGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::opencl::current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_devices);
  }
  std::call_once(opencl_gens_init_flag[idx], [&] {
    default_gens_opencl[idx] = std::make_shared<OpenCLGenerator>(idx);
    default_gens_opencl[idx]->seed();
  });
  return default_gens_opencl[idx].get();
}

/**
 * Utility to create a OpenCLGenerator. Returns a shared_ptr
 */
std::shared_ptr<OpenCLGenerator> createOpenCLGenerator(DeviceIndex device_index) {
  std::call_once(num_device_init_flag, initOpenCLGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::opencl::current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_devices, "The device_index is invalid.");
  auto gen = std::make_shared<OpenCLGenerator>(idx);
  gen->set_current_seed(default_rng_seed_val);
  gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace detail
} // namespace opencl

/**
 * OpenCLGenerator class implementation
 */
OpenCLGenerator::OpenCLGenerator(DeviceIndex device_index)
  : Generator{Device(DeviceType::OPENCL, device_index)} { }

/**
 * Sets the seed to be used by clrandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 * 
 * See Note [Acquire lock when using random generators]
 */
void OpenCLGenerator::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

/**
 * Gets the current seed of OpenCLGenerator.
 */
uint64_t OpenCLGenerator::current_seed() const {
  return seed_;
}

/**
 * Gets a nondeterminstic random number from /dev/urandom or time,
 * seeds the CPUGenerator with it and then returns that number.
 * 
 * FIXME: You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and CUDA (and OpenCL)
 */
uint64_t OpenCLGenerator::seed() {
  auto random = detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/**
 * Sets the philox_offset_per_thread_ to be used by clrandStatePhilox4_32_10
 * 
 * See Note [Acquire lock when using random generators]
 */
void OpenCLGenerator::set_philox_offset_per_thread(uint64_t offset) {
  philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of OpenCLGenerator.
 */
uint64_t OpenCLGenerator::philox_offset_per_thread() {
  return philox_offset_per_thread_;
}

/**
 * Gets the seed and philox offset value to be used in
 * clrandStatePhilox4_32_10
 * 
 * Each kernel using philox has to sensibly increment offset
 * for future users of philox. So it gets the "old" value for
 * itself (before add), and tells subsequent users which offset
 * they should use, since only the kernel knows how many randoms
 * it intends to generate. 
 * 
 * Increment should be at least the number of clrand() random numbers used in
 * each thread. It is the user's responsibility to make sure that the increment
 * for philox is never smaller than the number of clrand() calls. Increment
 * value > the number of clrand() calls won't harm but anything less would mean
 * that you would be reusing random values from previous calls.
 * 
 * See Note [Acquire lock when using random generators]
 */
std::pair<uint64_t, uint64_t> OpenCLGenerator::philox_engine_inputs(uint64_t increment) {
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
}

/*
 * Gets the DeviceType of OpenCLGenerator.
 * Used for type checking during run time.
 */
DeviceType OpenCLGenerator::device_type() {
  return DeviceType::OPENCL;
}

/**
 * Public clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<OpenCLGenerator> OpenCLGenerator::clone() const {
  return std::shared_ptr<OpenCLGenerator>(this->clone_impl());
}

/**
 * Private clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
OpenCLGenerator* OpenCLGenerator::clone_impl() const {
  auto gen = new OpenCLGenerator(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

} // namespace at

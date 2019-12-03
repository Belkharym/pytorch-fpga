# Dataflow

[![PyTorch Data Flow and Interface Diagram](https://raw.githubusercontent.com/wiki/pytorch/pytorch/images/pytorch_wiki_dataflow_interface_diagram.png)](https://github.com/pytorch/pytorch/wiki/PyTorch-Data-Flow-and-Interface-Diagram)

# [C10 Architecture](https://github.com/pytorch/pytorch/wiki/Software-Architecture-for-c10)


# C10

C10 is the core library of Pytorch. It contains the implementation of the `Tensor` class, the *new* dispatcher, and many miscellaneous things.\
For our purpose, the purpose of this library is to be a resource manager. It initialize the low level backends and communicate with it. It provides an API to the rest of Pytorch libraries (ATen and Caffe2) to allow them to use those backends.

The OpenCL implementation of the c10 library is very heavily based on the CUDA implementation. (We litterally copy-pasted the files from `c10/cuda/`)

The next sub sections will discribe the content of each files under `c10/opencl/` .

## `OpenCLMacros.h`

In addition to defining many of the basic macros, it is also the entry point of OpenCL in the code. This is the file where we setup which version of OpenCL we use and how we use it. (OpenCL 1.2, exceptions disabled)

## `OpenCLException[.cpp/.h]`

These files contains error handling macros and functions.

TODO : List and explain every macro.

## `OpenCLFunctions[.cpp/.h]`

This file is the file that provides an indirect access to the OpenCL API. There is a lazy initialization of OpenCL which is called when we first try to access the API.

## `OpenCLStream[.cpp/.h]`

This file is a specialisation of the `c10::Stream` class for OpenCL "streams". (The equivalent of a stream in OpenCL is a `cl::CommandQueue`)

## `OpenCLGuard.h` and `impl/OpenCLGuardImpl[.cpp/.h]`

This is something analogous to the STL's `std::lock_guard` for `std::mutex`, but instead of guarding the locking and unlocking of a mutex, it guards the current device. When you create an OpenCLGuard, it fetch the current device id and keeps it. Then, it sets the current device id to be the one given as a parameter to the constructor. When the OpenCLGuard is destroyed, it sets the current device id to be the original one.

## `OpenCLCachingAllocator[.cpp/.h]`

Unused.

# ATen

This folder contains the implementations of every operations.

## ATen/native

This folder contains the low level implementations of every operations that can be applied on Tensors.

Currently, there are 2 ways an operation can be registered in the dispatcher: The legacy way (directly through `ATen/native/native_functions.yaml`), and the new way (using the macro `REGISTER_DISPATCH(name, func)`)

#### Legacy registration

The legacy registration method passes through adding a new entry in the dispatch dictionary of the operation in `ATen/native/native_functions.yaml` .
Here is an example for the `and` operation:
```yaml
- func: __and__.Tensor(Tensor self, Tensor other) -> Tensor
  use_c10_dispatcher: full
  variants: method, function
  dispatch:
    CPU: legacy::cpu::_th_and
    CUDA: legacy::cuda::_th_and
    OpenCL: _and_opencl
```

This allows take care of the declaration of the function signature and of the registration of the function in the dispatcher.

Then, the implementation of the function must be defined in the `at::native` namespace inside of a `.cpp` file in the `ATen/native/opencl/` folder.
> NOTE: All the files under `ATen/native/opencl/` are automatically added to the source of the our shared library `libaten-opencl.so` when running `cmake`. You will need to re-run `cmake` when you add a new file to the folder. The inclusion of the folder is defined inside `ATen/CMakeLists.txt` .

#### New registration

The new registration method is very similar to the legacy one. For the operations that use the new dispatch registration, the dispatch dictionnary should have the same function assigned for both `CPU` and `CUDA`, and an other function for the sparse version of these:
```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  use_c10_dispatcher: full
  variants: function, method
  dispatch:
    CPU: add
    CUDA: add
    SparseCPU: add_sparse
    SparseCUDA: add_sparse
    MkldnnCPU: mkldnn_add
    OpenCL: add
  supports_named_tensor: True
```
Sometimes, there won't even be a dispatch dictionnary:
```yaml
- func: copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
  use_c10_dispatcher: unboxed_only
  variants: method
  device_guard: False
  supports_named_tensor: True
```
For this case, the dispatch is the same, wether the tensor is sparse or not.

In the first case, you only have to add an entry in the `dispatch` dictionnary for the Backend for which you want to add an operation.\
For the second case, you don't have to modify the `.yaml` file at all.

In any case, you have to implement in a file from `ATen/native/opencl/` a *kernel* function for the operation. Afterward, you can simply register the kernel with the macro `REGISTER_DISPATCH(add_stub, add_kernel_opencl);` (usually placed at the end of the file).

> NOTE: The best example for the use-case of the new registration method is in the file `ATen/native/opencl/OpenCLComparison.cpp`

The next few sections will describe what each file in the `ATen/native/opencl/` folder contains. Every file was inspired from an similarly-named file in the folder `ATen/native/cuda/` , if you need references to understand how to add new operations.

### `BinaryOpsKernel`

This file contains basic mathematical opearations applied element-wise and taking 2 operands. You can currently find the operations add, sub, div, mul, atan2, logical_xor.

### `Copy`

This file contains the implementation for the copy operation. This operation allows to copy from one tensor to an other, with or without a cast, between any OpenCL device and the CPU or between any two devices, synchronously or asynchronously.

### `FillKernel`

This file contains the `fill` operation which fills a Tensor with a given value.

### `MathBlas`

This file contains BLAS (Basic Linear Algebra Subsystem) operations.
Currently we don't have an implementation for any of these functions. We just redirect them to CUDA if it is available, or to the CPU otherwise.

### `OpenCLComparison`

This file contains the implementation of comparison operations, like `eq` (equal) or `lt` (less than).

### `OpenCLScalar`

Based off of `native/cuda/CUDAScalar.cu` . The function implemented in this file is used when we call the Tensor method `tensor.item()`. It is only usable on *scalar* tensors (tensors of dimentionality 0 containing only 1 element).

### `OpenCLTensor`

This is a utility file for tensors which mimics the legacy implementation inside of `THC`. (might be unnecessary after the merge of pytorch and caffe2)

### `Resize`

This is a utility file for resizing tensors which mimics the legacy implementation inside of `THC`. (might be unnecessary after the merge of pytorch and caffe2)

### `TensorFactories`

This file (currently) contains advanced operations. (*See the NOTE at the end of this sub section*)\
You can find the list here: 
- empty
- uniform
- random
- normal
- abs
- and
- masked
- ceil
- zero
- min
- max
- set
- cat
- remainder

> NOTE: This file should not contain all of these functions. This file should only contain function related to producing/modifying tensors (e.g. `empty`, `uniform`, `random`, `normal`, `zero`, `set`, `cat`). This file need a refactoring (for example, `and` should go in `BinaryOps`).

# Caffe2



# run locally

> NOTE: Even though Pytorch technically supports Python 2, it is strongly recommended to use Python 3, since Python 2 will be officially discontinued starting from January 2020.

```bash
python3 -m pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing

git submodule sync
git submodule update --init --recursive

env CMAKE_BUILD_TYPE=RelWithDebInfo PYTHON_EXECUTABLE=$(which python3) USE_CUDA=0 USE_OPENCL=ON python3 -m pip install --user -v -e .
```

# pipeline to run on aws instance


The pipeline to compile pytorch on aws instance is pretty tricky. The build instance memory resources (60 GB RAM + 20 GB Swap) is insufficient to finish the synthesis. Thus, the bitstream can not be generated.

So, we have to use this pipeline : 

1. generate bitstream of pytorch on F1 instance

    ```bash
    python3 -m pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing

    git submodule sync
    git submodule update --init --recursive

    env CMAKE_BUILD_TYPE=Release PYTHON_EXECUTABLE=$(which python3) USE_CUDA=0 USE_OPENCL=ON USE_FPGA=ON python3 -m pip install --user -v -e .
    ```

2. Send the `.xclbin` file to a build instance. Use a build instance with SDAccel 2018.3 .
   >NOTE: The `.xclbin` should be in the folder `<pytorch root>/torch/opencl/` if you are building from the `fpga` branch of the repository, and in `<pytorch root>/torch/opencl/kernels/` if you are using a more up-to-date branch.

3. Generate the AFI

    Here is a macro you can put inside a bash script and run in the same directory as the `.xclbin` file to generate a `.awsxclbin` file.
    ```bash
    makeAfi() {
        XCLBIN_FILE=$(find "${XCLBIN_DIR}" -type f -name '*.xclbin' 2>/dev/null | head -n 1)
        if [ "${RELEASE_VER}" == "" ] ; then
            echo "run sdaccel_setup.sh"
            source "${AWS_DIR}/sdaccel_setup.sh"
        fi
        if [ "${XCLBIN_FILE}" == "" ]; then
            echo 'ERROR MakeAfi : XCLBIN not found'
            echo "find in ${XCLBIN_DIR}"
        else
            echo 'XCLBIN file present ('"${XCLBIN_FILE}"')'
            echo 'Requesting AFI...'
            mkdir -p "${BINARY_DIR}"
            pushd "${BINARY_DIR}"
            local AWSXCLBIN_FILE=$(basename "${XCLBIN_FILE}" .xclbin)
            "${AWS_FPGA_REPO_DIR}/SDAccel/tools/create_sdaccel_afi.sh" \
                -s3_bucket="${S3_BUCKET}" \
                -s3_dcp_key="${S3_DCP_DIR}" \
                -s3_logs_key="${S3_LOGS_DIR}" \
                -xclbin="${XCLBIN_FILE}" \
                -o="${AWSXCLBIN_FILE}" && popd
        fi
    }
    ```

4. Send the `.awsxclbin` file back to the F1 instance where you compiled Pytorch. Replace the old `.xclbin` file by the new `.awxsclbin` .

5. Run Pytorch
    ```bash
    python3 -c "import torch; ocl = torch.device('opencl'); x = torch.ones(3,3,device=ocl); print(x)"
    ```

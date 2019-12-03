# Dataflow

[![PyTorch Data Flow and Interface Diagram](https://raw.githubusercontent.com/wiki/pytorch/pytorch/images/pytorch_wiki_dataflow_interface_diagram.png)](https://github.com/pytorch/pytorch/wiki/PyTorch-Data-Flow-and-Interface-Diagram)

# [C10 Architecture](https://github.com/pytorch/pytorch/wiki/Software-Architecture-for-c10)

---

# C10

C10 is the core library of PyTorch. It contains the implementation of the `Tensor` class, the *new* dispatcher, and many miscellaneous things.\
For our purpose, the purpose of this library is to be a resource manager. It initialize the low level backends and communicate with it. It provides an API to the rest of PyTorch libraries (ATen and Caffe2) to allow them to use those backends.

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

### Operations Registration

This folder contains the low level implementations of every operations that can be applied on Tensors.

Currently, there are 2 ways an operation can be registered in the dispatcher: The legacy way (directly through `ATen/native/native_functions.yaml`), and the new way (using the macro `REGISTER_DISPATCH(name, func)`)

#### Legacy registration

The legacy registration method passes through adding a new entry in the dispatch dictionary of the operation in `ATen/native/native_functions.yaml`.\
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
> <span style="color:#ffcc00">ℹ️</span> All the `.cpp`/`.h` files under `ATen/native/opencl/` are automatically added to the source of our shared library `libaten-opencl.so` when running `cmake`. You will need to re-run `cmake` when you add a new file to the folder. The inclusion of the folder is defined inside `ATen/CMakeLists.txt`.

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

> <span style="color:#ffcc00">ℹ️</span> The best example for the use-case of the new registration method is in the file `ATen/native/opencl/OpenCLComparison.cpp`

### OpenCL Methodology

Since this repository is about having **PyTorch** running on FPGAs, we had to take the limitiations of those FPGAs in consideration. One of the limitations is that there is a limited amount of entry points the FPGA can support. (An entry point correspond to an **OpenCL** kernel.)

To work around that limitation, we decided to have a kernel for each function signature, and pass the operation to apply as parameter. To reduce further the number of kernels, we get the buffers as `void *` in the kernels, and we pass the types of the data contained in the buffers as one of the parameters.

## Listing of `ATen/native/opencl`

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

This is a utility file for tensors which mimics the legacy implementation inside of `THC`. (might be unnecessary after the merge of PyTorch and caffe2)

### `Resize`

This is a utility file for resizing tensors which mimics the legacy implementation inside of `THC`. (might be unnecessary after the merge of PyTorch and caffe2)

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

> <span style="color:#ffcc00">ℹ️</span> This file should not contain all of these functions. This file should only contain function related to producing/modifying tensors (e.g. `empty`, `uniform`, `random`, `normal`, `zero`, `set`, `cat`). This file need a refactoring (for example, `and` should go in `BinaryOps`).

# Caffe2

Currently, **Caffe2** is the library that handles the highlevel calculations of the graph used to automatically calculate the gradiant of every operation applied to a tensor.

For our purpose, this library contains the file `caffe2/opencl/context[.cc/.h]` which sole purpose is to provide an `Allocator` to allocate memory on the devices, and to provide basic data transfer functions.

## The `torch` folder

We also added a few files to the `torch` folder, but it was mainly to give a similar user experience from how they can use **CUDA**.\
We mainly added files to give access to functionalities to the user. For example `torch.opencl.is_available()`, `torch.opencl.device_count()`, `torch.opencl.get_device_name(deviceId)`, or `torch.opencl.get_device_properties(deviceId)`.

# Run locally

> <span style="color:#ffcc00">ℹ️</span> Even though **PyTorch** technically supports **Python 2**, it is strongly recommended to use **Python 3**, since **Python 2** will be [officially discontinued](https://pip.pypa.io/en/latest/development/release-process/#python-2-support) starting from January 2020.

To compile and install **PyTorch** in developpement mode, you can run the following commands inside a bash terminal located under the root directory of **PyTorch**'s repository:
```bash
python3 -m pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing

git submodule sync
git submodule update --init --recursive

env CMAKE_BUILD_TYPE=RelWithDebInfo PYTHON_EXECUTABLE=$(which python3) USE_CUDA=0 USE_OPENCL=ON python3 -m pip install --user -v -e .
```

The 2 first sections of this `bash` script are fairly straight forward:

The first line installs the packages required to *build* PyTorch. (The ones needed only to run are in the file `requirements.txt`)

The 2 next lines synchronize and initialize the git submodules used by PyTorch.

The third line is a little more complex. So let's deconstruct it:
- `env [OPTION]... [-] [NAME=VALUE]... [COMMAND [ARG]...]`: It's a `bash` command that sets environment variables, but only for the scope of the execution of the given command.
- `CMAKE_BUILD_TYPE=RelWithDebInfo`: The build type `RelWithDebInfo` is equivalent to have the project compiled with optimisations (with the compile option `-O2`), but while keeping the debug information (with the compile opetion `-g`). Since a full build with a build type `Debug` is way longer than `Release` or `RelWithDebInfo`, this allows to debug the code while not taking too much time to compile.
- `PYTHON_EXECUTABLE=$(which python3)`: By default, if **Python 2** is installed on the system, **PyTorch** will compile for **Python 2**. However, as stated above, **Python 2** will be/has been [discontinued](https://pip.pypa.io/en/latest/development/release-process/#python-2-support) starting from January 2020. This is why we use `python3`.
- `USE_CUDA=0 USE_OPENCL=ON`: It basically does as it reads. It disables the **CUDA** implementation and enables the **OpenCL** one.
- `python3 -m pip install --user -v -e .`: To build and install pytorch, we use the **Python** package manager **Pip**. There are 3 options we provide:
  - `--user`: Specifies that we want to install it for the current user only, instead of globally. We don't want to install **PyTorch** globally when developping, since there can be some permission issues.
  - `-v`: For verbose output. This allows us to get the output logs of the build.
  - `-e .`: This option force `pip` to install **PyTorch** inplace. This mean that the results of the build will be installed directly inside the repository. There will still be 1 file installed in `pip`'s registry, but it is only to refer to the location of the repository where **PyTorch** was installed.

# Pipeline to run on aws instance


The pipeline to compile PyTorch on aws instance is pretty tricky. The build instance memory resources (60 GB RAM + 20 GB Swap) is insufficient to finish the synthesis. Thus, the bitstream can not be generated.

So, we have to use this pipeline : 

#### Prerequisites

You first need to make sure **Python 3** is installed on the system.
You might need to also install **OpenSSL** manually if **Python 3** is not on the system.

```bash
setupOpenSSL() {
    if [ $(python -c "import sys; print(sys.version_info[0])") -eq "3" ]; then
        sudo -s ln -fs $(which python2) $(which python)
        sudo -s yum groupinstall -y "Developement Tools"
        sudo -s ln -fs $(which python3) $(which python)
    else
        sudo -s yum groupinstall -y "Developement Tools"
    fi
    wget https://ftp.openssl.org/source/old/1.1.1/openssl-1.1.1.tar.gz

    tar xvf openssl-1.1.1.tar.gz
    rm -f openssl-1.1.1.tar.gz
    cd openssl-1.1.1
    ./config --prefix=/usr --openssldir=/etc/ssl --libdir=lib no-shared zlib-dynamic
    make
    sudo -s make install
    cd ..

    export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64
    ldconfig
    openssl version
}
setupOpenSSL
```

You can then use the `yum` package manager to install **Python 3**:

```bash
sudo yum install -y python3 python3-pip
```

1. Generate bitstream of PyTorch on F1 instance

    ```bash
    python3 -m pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing

    git submodule sync
    git submodule update --init --recursive

    env CMAKE_BUILD_TYPE=Release PYTHON_EXECUTABLE=$(which python3) USE_CUDA=0 USE_OPENCL=ON USE_FPGA=ON python3 -m pip install --user -v -e .
    ```

2. Send the `.xclbin` file to a build instance. Use a build instance with SDAccel 2018.3 .
   > <span style="color:#ffcc00">ℹ️</span> The `.xclbin` should be in the folder `<PyTorch root>/torch/opencl/` if you are building from the `fpga` branch of the repository, and in `<PyTorch root>/torch/opencl/kernels/` if you are using a more up-to-date branch.

3. Generate the AFI

    Here is a macro you can put inside a bash script, and you can run it in the same directory as the `.xclbin` file to generate a `.awsxclbin` file.
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

4. Send the `.awsxclbin` file back to the F1 instance where you compiled PyTorch. Replace the old `.xclbin` file by the new `.awxsclbin` .

5. Run PyTorch
    ```bash
    python3 -c "import torch; ocl = torch.device('opencl'); x = torch.ones(3,3,device=ocl); print(x)"
    ```

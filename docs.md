# Dataflow

https://github.com/pytorch/pytorch/wiki/PyTorch-Data-Flow-and-Interface-Diagram

# C10 Architecture

https://github.com/pytorch/pytorch/wiki/Software-Architecture-for-c10   


# C10


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
For the second case, you don't have to modify the `.yaml` file.

In any case, you have to implement in a file from `ATen/native/opencl/` a *kernel* function for the operation. Afterward, you can simply register the kernel with the macro `REGISTER_DISPATCH(add_stub, add_kernel_opencl);` (usually placed at the end of the file).

> NOTE: The best example for the use-case of the new registration method is in the file `ATen/native/opencl/OpenCLComparison.cpp`

The next few sections will describe what each file in the `ATen/native/opencl/` folder contains. Every file was inspired from an similarly-named file in the folder `ATen/native/cuda/` , if you need references to understand how to add new operations.

### BinaryOpsKernel

This file contains some simple functions of math. Actually, you can find function to add, sub, div, mul, atan2, logical_xor

### Copy

This file contains the implementation for the copy operation. This operation allows to copy from one tensor to an other, with or without a cast, between any OpenCL device and the CPU or between any two devices, synchronously or asynchronously.

### FileKernel

This file contains one functions which let to fill a Tensor in device with a value

### MathBlas

This file contains some functions to call when you want to use MathBlas. MathBlas is a library that optimizes linear algebra calculations.
Currently we don't have implement any functions. We just have dispatch them to cuda if it's available or cpu.

### OpenCLComparison

This file contains some functions of comparisation, as equal.

### OpenCLScalar

Based off of `native/cuda/CUDAScalar.cu` . The function implemented in this file is used when we call the Tensor method `tensor.item()`. It is only usable on *scalar* tensors (tensors of dimentionality 0 containing only 1 element).

### OpenCLTensor

This is a utility file for tensors which mimics the legacy implementation inside of `THC`. (might be unnecessary after the merge of pytorch and caffe2)

### Resize

This is a utility file for resizing tensors which mimics the legacy implementation inside of `THC`. (might be unnecessary after the merge of pytorch and caffe2)

### TensorFactories

This file contains advanced operations.
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


```
python3 -m pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing

git submodule sync
git submodule update --init --recursive

USE_CUDA=0 USE_OPENCL=ON python3 ./setup.py install
```

# pipeline to run on aws instance


The pipeline to compile pytorch on aws instance is pretty tricky. The build instance does not have the amount of ram (60 GB RAM + 20 GB swap) needed to finish the synthese. The bitstream can not be generated.

So, use this pipeline : 

1. generate bitstream of pytorch on F1 instance

```
python3 -m pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing

git submodule sync
git submodule update --init --recursive

USE_CUDA=0 USE_OPENCL=ON USE_FPGA=ON  $(which python3) python3 ./setup.py install
```

2. Send the [name_project].xclbin to build instance. Use a build instance with SDAccel 2018.3 

3. Generate the AFI

```
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

4. Send the [nameproject].awsxclbin to F1 instance

5. Run Pytorch

You can test any test file to check if pytorch is functionnal.
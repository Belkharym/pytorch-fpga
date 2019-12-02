# Dataflow

https://github.com/pytorch/pytorch/wiki/PyTorch-Data-Flow-and-Interface-Diagram

# C10 Architecture

https://github.com/pytorch/pytorch/wiki/Software-Architecture-for-c10   


# C10


# ATen

ATen has some calcul functions.

## BinaryOpsKernel

This file contains some simple functions of math. Actually, you can find function to add, sub, div, mul, atan2, logical_xor

## Copy

This file contains one functions which let to copy a TensorIterator in CPU to device

## FileKernel

This file contains one functions which let to fill a Tensor in device with a value

## MathBlas

This file contains some functions to call when you want to use MathBlas. MathBlas is a library that optimizes linear algebra calculations.
Currently we don't have implement any functions. We just have dispatch them to cuda if it's available or cpu.

## OpenCLComparison

This file contains some functions of comparisation, as equal.

## OpenCLScalar

## OpenCLTensor

This file contains some functions to apply to Tensor.

## Resize

This file contains some functions to let user resize a Tensor in the memory

## TensorFactories

This file contains advances calcul function. 
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
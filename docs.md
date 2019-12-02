# Dataflow

https://github.com/pytorch/pytorch/wiki/PyTorch-Data-Flow-and-Interface-Diagram

# C10 Architecture

https://github.com/pytorch/pytorch/wiki/Software-Architecture-for-c10   


# C10


# ATen


# Caffe2



# run locally


```
python3 -m pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing

git submodule sync
git submodule update --init --recursive

USE_CUDA=0 USE_OPENCL=ON python3 ./setup.py install
```

# pipeline to run on aws instance


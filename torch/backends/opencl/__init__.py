import sys
import torch


def is_built():
    r"""Returns whether PyTorch is built with CUDA support.  Note that this
    doesn't necessarily mean CUDA is available; just that if this PyTorch
    binary were run a machine with working CUDA drivers and devices, we
    would be able to use it."""
    return torch._C.has_opencl


class OpenCLModule(object):
    def __init__(self, m):
        self.__dict__ = m.__dict__
        # You have to retain the old module, otherwise it will
        # get GC'ed and a lot of things will break.  See:
        # https://stackoverflow.com/questions/47540722/how-do-i-use-the-sys-modules-replacement-trick-in-init-py-on-python-2
        self.__old_mod = m

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = OpenCLModule(sys.modules[__name__])

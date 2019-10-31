r"""
This package adds support for OpenCL tensor types, that implement the same
function as CPU tensors, but they utilize external Devices (like (integrated)
GPUs or FPGA boards) for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports OpenCL.

:ref:`opencl-semantics` has more details about working with OpenCL.
"""

import contextlib
import platform
import ctypes
import os
import sys
import torch
import traceback
import warnings
import threading
from torch._six import raise_from
from subprocess import Popen, PIPE
from multiprocessing.util import register_after_fork as _register_after_fork
from ._utils import _get_device_index

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_in_bad_fork = False  # this global is also used in torch.manual_seed
_original_pid = False

# NOTE The OpenCL library should be under the same path as CUDA on windows if cuda is present.
# TODO Check on a windows machine if it is viable
def find_opencl_windows_lib():
    # Override the default search process
    # Fixes https://github.com/pytorch/pytorch/issues/20202
    # The libary selection will be done in these directories one by one
    # 1. [Package Root]\Lib
    #    That's where our libraries are in, which should be loaded first.
    # 2. [Python Root]\Library\bin
    #    That's where `cudatoolkit` store the cuda libraries.
    # 3. Default directories
    #    That is stored in the environment variable `PATH`.
    test_env = os.environ.copy()
    old_path = test_env['PATH']
    py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
    th_dll_path = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'lib')
    test_env['PATH'] = ';'.join([th_dll_path, py_dll_path, old_path])
    proc = Popen(['where', 'OpenCL*.dll'], stdout=PIPE,
                 stderr=PIPE, stdin=PIPE, env=test_env)
    out, err = proc.communicate()
    out = out.decode().strip()
    if len(out) > 0:
        if out.find('\r\n') != -1:
            out = out.split('\r\n')[0]
        cuda_lib_name = os.path.basename(out)
        cuda_lib = os.path.splitext(cuda_lib_name)[0]
        cuda_lib = str(cuda_lib)
        return ctypes.cdll.LoadLibrary(cuda_lib)
    else:
        return None


def is_available():
    r"""Returns a bool indicating if OpenCL is currently available."""
    if (not hasattr(torch._C, '_opencl_isDriverSufficient') or
            not torch._C._opencl_isDriverSufficient()):
        return False
    return torch._C._opencl_getDeviceCount() > 0


def _sleep(cycles):
    torch._C._opencl_sleep(cycles)


def _check_driver():
    if not hasattr(torch._C, '_opencl_isDriverSufficient'):
        raise AssertionError("Torch not compiled with OpenCL enabled")
    if not torch._C._opencl_isDriverSufficient():
        if torch._C._opencl_getDriverVersion() == 0:
            # found no OpenCL driver on the system
            raise AssertionError("""
Found no OpenCL driver on your system. Please check that you
have an OpenCL Device and installed a driver from your vendor (
http://www.nvidia.com/Download/index.aspx for example)""")
        else:
            # TODO: directly link to the alternative bin that needs install
            raise AssertionError("""
The OpenCL driver on your system is too old (found version {}).
Please update your Device driver by downloading and installing a new
version from the URL: http://www.nvidia.com/Download/index.aspx for NVIDIA GPUs
Alternatively, go to: https://pytorch.org to install
a PyTorch version that has been compiled with your version
of the OpenCL driver.""".format(str(torch._C._opencl_getDriverVersion())))


def _check_capability():
    incorrect_binary_warn = """
    Found Device%d %s which requires OPENCL_VERSION >= %d to
     work properly, but your PyTorch was compiled
     with OPENCL_VERSION %d. Please install the correct PyTorch binary
     using instructions from https://pytorch.org
    """

    old_device_warn = """
    Found Device%d %s which is of opencl capability %d.%d.
    PyTorch no longer supports this Device because it is too old.
    The minimum opencl capability that we support is 1.2.
    """

    OPENCL_VERSION = torch._C._opencl_getCompiledVersion()
    for d in range(device_count()):
        capability = get_device_capability(d)
        major = capability[0]
        minor = capability[1]
        name = get_device_name(d)
        if capability == (1, 0) or (major < 1 and minor < 2):
            warnings.warn(old_device_warn % (d, name, major, capability[1]))
        elif OPENCL_VERSION <= 200 and major >= 2 and minor >= 0:
            warnings.warn(incorrect_binary_warn % (d, name, 120, OPENCL_VERSION))


def _lazy_call(callable):
    if _initialized:
        callable()
    else:
        # Don't store the actual traceback to avoid memory cycle
        _queued_calls.append((callable, traceback.format_stack()))

_lazy_call(_check_capability)


class DeferredOpenCLCallError(Exception):
    pass


def init():
    r"""Initialize PyTorch's OpenCL state.  You may need to call
    this explicitly if you are interacting with PyTorch via
    its C API, as Python bindings for OpenCL functionality will not
    be until this initialization takes place.  Ordinary users
    should not need this, as all of PyTorch's OpenCL methods
    automatically initialize OpenCL state on-demand.

    Does nothing if the OpenCL state is already initialized.
    """
    _lazy_init()


def _lazy_init():
    global _initialized, _original_pid, _queued_calls
    if _initialized or hasattr(_tls, 'is_initializing'):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if _initialized:
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _in_bad_fork:
            from sys import version_info
            if version_info < (3, 4):
                msg = ("To use OpenCL with multiprocessing, you must use Python "
                       "3.4+ and the 'spawn' start method")
            else:
                msg = ("To use OpenCL with multiprocessing, you must use the "
                       "'spawn' start method")
            raise RuntimeError(
                "Cannot re-initialize OpenCL in forked subprocess. " + msg)
        _check_driver()
        torch._C._opencl_init()
        _original_pid = os.getpid()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True
        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = ("OpenCL call failed lazily at initialization with error: {}\n\n"
                           "OpenCL call was originally invoked at:\n\n{}").format(str(e), orig_traceback)
                    raise_from(DeferredOpenCLCallError(msg), e)
        finally:
            delattr(_tls, 'is_initializing')
        _initialized = True


def _after_fork(arg):
    global _initialized, _in_bad_fork
    if _initialized and _original_pid != os.getpid():
        _initialized = False
        _in_bad_fork = True
        _OpenCLBase.__new__ = _lazy_new
        torch._C._opencl_set_run_yet_variable_to_false()

_register_after_fork(_after_fork, _after_fork)


class openclStatus(object):
    SUCCESS = 0
    ERROR_NOT_READY = 34


class OpenCLError(RuntimeError):
    def __init__(self, code):
        super(OpenCLError, self).__init__('({1})'.format(code))


def check_error(res):
    if res != openclStatus.SUCCESS:
        raise OpenCLError(res)


class device(object):
    r"""Context-manager that changes the selected device.

    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = torch._C._opencl_getDevice()
        if self.prev_idx != self.idx:
            torch._C._opencl_setDevice(self.idx)
        _lazy_init()

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch._C._opencl_setDevice(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a Device, this is a no-op.

    Arguments:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_opencl else -1
        super(device_of, self).__init__(idx)


def set_device(device):
    r"""Sets the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``OPENCL_VISIBLE_DEVICES`` environmental variable.

    Arguments:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        torch._C._opencl_setDevice(device)


def get_device_name(device=None):
    r"""Gets the name of a device.

    Arguments:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.opencl.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return get_device_properties(device).name


def get_device_capability(device=None):
    r"""Gets the opencl capability of a device.

    Arguments:
        device (torch.device or int, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.opencl.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor opencl capability of the device
    """
    prop = get_device_properties(device)
    versions = prop.version.split(" ")[1].split(".")
    major = int(versions[0])
    minor = int(versions[1])
    return major, minor


def get_device_properties(device):
    if not _initialized:
        init()  # will define _get_device_properties and _OpenCLDeviceProperties
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _get_device_properties(device)


@contextlib.contextmanager
def stream(stream):
    r"""Context-manager that selects a given stream.

    All OpenCL kernels queued within its context will be enqueued on a selected
    stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.

    .. note:: Streams are per-device. If the selected stream is not on the
        current device, this function will also change the current device to
        match the stream.
    """
    if stream is None:
        yield
        return
    src_prev_stream = current_stream()

    if src_prev_stream.device != stream.device:
        # The given stream is on a different device; have to restore the
        # current_stream on that device on exit as well
        with device(stream.device):
            dst_prev_stream = current_stream()

    torch._C._opencl_setStream(stream._cdata)
    try:
        yield
    finally:
        if src_prev_stream.device != stream.device:
            torch._C._opencl_setStream(dst_prev_stream._cdata)
        torch._C._opencl_setStream(src_prev_stream._cdata)


def device_count():
    r"""Returns the number of Devices available."""
    if is_available():
        return torch._C._opencl_getDeviceCount()
    else:
        return 0


def current_device():
    r"""Returns the index of a currently selected device."""
    _lazy_init()
    return torch._C._opencl_getDevice()


def synchronize(device=None):
    r"""Waits for all kernels in all streams on a OpenCL device to complete.

    Arguments:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.opencl.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    with torch.opencl.device(device):
        return torch._C._opencl_synchronize()


def current_stream(device=None):
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.opencl.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    return torch.opencl.Stream(_cdata=torch._C._opencl_getCurrentStream(
        _get_device_index(device, optional=True)))


def default_stream(device=None):
    r"""Returns the default :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.opencl.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    return torch.opencl.Stream(_cdata=torch._C._opencl_getDefaultStream(
        _get_device_index(device, optional=True)))


def empty_cache():
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other Device application and visible in
    `nvidia-smi`.

    .. note::
        :func:`~torch.opencl.empty_cache` doesn't increase the amount of Device
        memory available for PyTorch. See :ref:`opencl-memory-management` for
        more details about Device memory management.
    """
    if _initialized:
        torch._C._opencl_emptyCache()


def memory_allocated(device=None):
    r"""Returns the current Device memory occupied by tensors in bytes for a given
    device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.opencl.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `nvidia-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on Device. See :ref:`opencl-memory-management` for more
        details about Device memory management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._opencl_memoryAllocated(device)


def max_memory_allocated(device=None):
    r"""Returns the maximum Device memory occupied by tensors in bytes for a given
    device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.opencl.reset_max_memory_allocated` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.opencl.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`opencl-memory-management` for more details about Device memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._opencl_maxMemoryAllocated(device)


def reset_max_memory_allocated(device=None):
    r"""Resets the starting point in tracking maximum Device memory occupied by
    tensors for a given device.

    See :func:`~torch.opencl.max_memory_allocated` for details.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.opencl.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`opencl-memory-management` for more details about Device memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._opencl_resetMaxMemoryAllocated(device)


def memory_cached(device=None):
    r"""Returns the current Device memory managed by the caching allocator in bytes
    for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.opencl.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`opencl-memory-management` for more details about Device memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._opencl_memoryCached(device)


def max_memory_cached(device=None):
    r"""Returns the maximum Device memory managed by the caching allocator in bytes
    for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.opencl.reset_max_memory_cached` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.opencl.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`opencl-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._opencl_maxMemoryCached(device)


def reset_max_memory_cached(device=None):
    r"""Resets the starting point in tracking maximum Device memory managed by the
    caching allocator for a given device.

    See :func:`~torch.opencl.max_memory_cached` for details.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.opencl.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`opencl-memory-management` for more details about Device memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._opencl_resetMaxMemoryCached(device)


def _host_allocator():
    _lazy_init()
    return torch._C._opencl_cudaHostAllocator()


@contextlib.contextmanager
def _free_mutex():
    torch._C._opencl_lock_mutex()
    try:
        yield
    finally:
        torch._C._opencl_unlock_mutex()


from .random import *

################################################################################
# Define Storage and Tensor classes
################################################################################


from ..storage import _StorageBase


def _dummy_type(name):
    def init_err(self):
        class_name = self.__class__.__name__
        raise RuntimeError(
            "Tried to instantiate dummy base class {}".format(class_name))
    return type(storage_name, (object,), {"__init__": init_err})


if not hasattr(torch._C, 'OpenCLDoubleStorageBase'):
    # Define dummy base classes
    for t in ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Half', 'Bool', 'BFloat16']:
        storage_name = 'OpenCL{0}StorageBase'.format(t)
        tensor_name = 'OpenCL{0}TensorBase'.format(t)

        torch._C.__dict__[storage_name] = _dummy_type(storage_name)
        torch._C.__dict__[tensor_name] = _dummy_type(tensor_name)

    torch._C.__dict__['_OpenCLStreamBase'] = _dummy_type('OpenCLStreamBase')
    torch._C.__dict__['_OpenCLEventBase'] = _dummy_type('OpenCLEventBase')


@staticmethod
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We need this method only for lazy init, so we can remove it
    del _OpenCLBase.__new__
    return super(_OpenCLBase, cls).__new__(cls, *args, **kwargs)


class _OpenCLBase(object):
    is_cuda = False
    is_opencl = True
    is_sparse = False

    def type(self, *args, **kwargs):
        with device(self.get_device()):
            return super(_OpenCLBase, self).type(*args, **kwargs)

    __new__ = _lazy_new


class DoubleStorage(_OpenCLBase, torch._C.OpenCLDoubleStorageBase, _StorageBase):
    pass


class FloatStorage(_OpenCLBase, torch._C.OpenCLFloatStorageBase, _StorageBase):
    pass


class LongStorage(_OpenCLBase, torch._C.OpenCLLongStorageBase, _StorageBase):
    pass


class IntStorage(_OpenCLBase, torch._C.OpenCLIntStorageBase, _StorageBase):
    pass


class ShortStorage(_OpenCLBase, torch._C.OpenCLShortStorageBase, _StorageBase):
    pass


class CharStorage(_OpenCLBase, torch._C.OpenCLCharStorageBase, _StorageBase):
    pass


class ByteStorage(_OpenCLBase, torch._C.OpenCLByteStorageBase, _StorageBase):
    pass


class HalfStorage(_OpenCLBase, torch._C.OpenCLHalfStorageBase, _StorageBase):
    pass


class BoolStorage(_OpenCLBase, torch._C.OpenCLBoolStorageBase, _StorageBase):
    pass


class BFloat16Storage(_OpenCLBase, torch._C.OpenCLBFloat16StorageBase, _StorageBase):
    pass

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)

from . import sparse  # noqa: F401
from . import profiler  # noqa: F401
from .streams import Stream, Event  # noqa: F401

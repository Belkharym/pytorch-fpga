from typing import Optional, Tuple, Union, List
from enum import Enum
import ctypes
from .. import device as _device

def is_available() -> bool: ...
def init() -> None: ...

class openclStatus:
    SUCCESS: int
    ERROR_NOT_READY: int

class OpenCLError:
    def __init__(self, code: int) -> None: ...

class _OpenCLDeviceFPConfig(Enum):
    """BitField"""
    denorm = 1
    inf_nan = 2
    round_to_nearest = 4
    round_to_zero = 8
    round_to_inf = 16
    fma = 32
    soft_float = 64
    correctly_rounded_divide_sqrt = 128
    def __repr__(self) -> str: ...

class _OpenCLDeviceExecCapabilities(Enum):
    """BitField"""
    kernel = 1
    native_kernel = 2

class _OpenCLDeviceProperties:
    addressBits: int
    available: bool
    builtInKernels: List[str]
    compilerAvailable: bool
    doubleFpConfig: int
    endianLittle: bool
    errorCorrectionSupport: bool
    executionCapabilities: int
    extensions: List[str]
    globalMemCacheSize: int
    globalMemCacheType: int
    globalMemCachelineSize: int
    globalMemSize: int
    halfFpConfig: int
    hostUnifiedMemory: bool
    imageSupport: bool
    image2dMaxHeight: int
    image2dMaxWidth: int
    image3dMaxDepth: int
    image3dMaxHeight: int
    image3dMaxWidth: int
    imageMaxBufferSize: int
    imageMaxArraySize: int
    linkerAvailable: bool
    localMemSize: int
    localMemType: int
    maxClockFrequency: int
    maxComputeUnits: int
    maxConstantArgs: int
    maxConstantBufferSize: int
    maxMemAllocSize: int
    maxParameterSize: int
    maxReadImageArgs: int
    maxSamplers: int
    maxWorkGroupSize: int
    maxWorkItemDimensions: int
    maxWorkItemSizes: List[int]
    maxWriteImageArgs: int
    memBaseAddrAlign: int
    minDataTypeAlignSize: int
    name: str
    nativeVectorWidthChar: int
    nativeVectorWidthDouble: int
    nativeVectorWidthFloat: int
    nativeVectorWidthHalf: int
    nativeVectorWidthInt: int
    nativeVectorWidthLong: int
    nativeVectorWidthShort: int
    openclCVersion: str
    parentDevice: int
    partitionMaxSubDevices: int
    partitionProperties: List[int]
    partitionAffinityDomain: int
    partitionType: List[int]
    platform: int
    preferredVectorWidthChar: int
    preferredVectorWidthDouble: int
    preferredVectorWidthFloat: int
    preferredVectorWidthHalf: int
    preferredVectorWidthInt: int
    preferredVectorWidthLong: int
    preferredVectorWidthShort: int
    printfBufferSize: int
    preferredInteropUserSync: bool
    profile: str
    profilingTimerResolution: int
    queueProperties: int
    referenceCount: int
    singleFpConfig: int
    type: int
    vendor: str
    vendorId: int
    version: str // Device version
    driverVersion: str // Driver version

_device_t = Union[_device, int]

def check_error(res: int) -> None: ...
def device_count() -> int: ...
def empty_cache() -> None: ...
def synchronize(device: _device_t) -> None: ...
def set_device(device: _device_t) -> None: ...
def get_device_capability(device: Optional[_device_t]=...) -> Tuple[int, int]: ...
def get_device_name(device: Optional[_device_t]=...) -> str: ...
def get_device_properties(device: _device_t) -> _OpenCLDeviceProperties: ...
def current_device() -> int: ...
def memory_allocated(device: Optional[_device_t]=...) -> int: ...
def max_memory_allocated(device: Optional[_device_t]=...) -> int: ...
def reset_max_memory_allocated(device: Optional[_device_t]=...) -> None: ...
def memory_cached(device: Optional[_device_t]=...) -> int: ...
def max_memory_cached(device: Optional[_device_t]=...) -> int: ...
def reset_max_memory_cached(device: Optional[_device_t]=...) -> None: ...
def find_cuda_windows_lib() -> Optional[ctypes.CDLL]: ...
def set_rng_state(new_state): ...
def get_rng_state(): ...

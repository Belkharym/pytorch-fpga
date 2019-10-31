import ctypes
import tempfile
import contextlib
from . import check_error


class openclOutputMode(object):
    openclKeyValuePair = ctypes.c_int(0)
    openclCSV = ctypes.c_int(1)

    @staticmethod
    def for_key(key):
        if key == 'key_value':
            return openclOutputMode.openclKeyValuePair
        elif key == 'csv':
            return openclOutputMode.openclCSV
        else:
            raise RuntimeError("supported OpenCL profiler output modes are: key_value and csv")

DEFAULT_FLAGS = [
    "devicestarttimestamp",
    "deviceendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "streamid",
    "enableonstart 0",
    "conckerneltrace",
]


def init(output_file, flags=None, output_mode='key_value'):
    flags = DEFAULT_FLAGS if flags is None else flags
    output_mode = cudaOutputMode.for_key(output_mode)
    with tempfile.NamedTemporaryFile(delete=True) as f:
        f.write(b'\n'.join(map(lambda f: f.encode('ascii'), flags)))
        f.flush()


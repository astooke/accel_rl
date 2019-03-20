
import numpy as np
import multiprocessing as mp
import ctypes


def np_mp_array(shape, dtype):
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    mp_array = mp.RawArray(ctypes.c_char, nbytes)
    return np.frombuffer(mp_array, dtype=dtype, count=size).reshape(shape)

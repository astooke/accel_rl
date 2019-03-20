
import numpy as np

from accel_rl.buffers.array import np_mp_array
from accel_rl.util.misc import struct


###############################################################################
#
# API
#
###############################################################################


def batch_buffer(example, length, shared=False):
    if isinstance(example, dict):
        buf = struct()
        for k, v in example.items():
            buf[k] = batch_buffer(v, length, shared)
    else:
        buf = build_array(example, length, shared)
    return buf


def build_array(value, length, shared=False):
    v = np.asarray(value)
    if v.dtype == "object":
        raise TypeError("Unsupported buffer example data type "
            "(can provide nested dictionaries, but final values must be "
            "able to cast under np.asarray() and not result in "
            "dtype=='object')")
    constr = np_mp_array if shared else np.zeros
    return constr(shape=(length,) + v.shape, dtype=v.dtype)


def buffer_with_segs_view(examples, length, segment_length, shared=False):
    buf = batch_buffer(examples, length, shared)
    buf.segs_view = view_segments(buf, segment_length)
    return buf


def buffer_length(batch_buffer):
    """
    Make sure the lengths of all data is the same, and return that length.
    """
    length = None
    for k, v in batch_buffer.items():
        # ipdb.set_trace()
        if k != "segs_view" and not k.startswith("extra"):
            if isinstance(v, struct):
                length = _recurse_buffer_length(v, length, [k])
            elif length is None:
                length = len(v)
            elif length != len(v):
                raise RuntimeError("Different lengths in buffer: {}".format(k))
    return length


def view_segments(batch_buffer, segment_length):
    length = buffer_length(batch_buffer)
    if length % segment_length != 0:
        raise ValueError("Buffer length ({}) not divisible by requested "
            "segment_length ({})".format(length, segment_length))
    num_segments = length // segment_length
    segments = list()
    i = 0
    for _ in range(num_segments):
        segment = struct()
        for k, v in batch_buffer.items():
            if isinstance(v, struct):
                segment[k] = _recurse_segment(v, i, segment_length)
            else:
                segment[k] = v[i:i + segment_length]
        segments.append(segment)
        i += segment_length
    return segments


def combine_distinct_buffers(buffer_1, buffer_2):
    """Assumes no terms match even in the first level of depth."""
    buf = buffer_1.copy()
    buffer_2 = buffer_2.copy()
    if hasattr(buf, "segs_view") and hasattr(buffer_2, "segs_view"):
        assert len(buf.segs_view) == len(buffer_2.segs_view)
        segs_2 = buffer_2.pop("segs_view")
        for seg1, seg2 in zip(buf.segs_view, segs_2):
            seg1.update(seg2)
    buf.update(buffer_2)
    return buf


def count_buffer_size(batch_buffer):
    size = 0
    for k, v in batch_buffer.items():
        if k != "segs_view":
            if isinstance(v, struct):
                size += count_buffer_size(v)
            else:
                size += v.nbytes  # (should only be numpy arrays at the botom)
    return size


###############################################################################
#
# Helpers
#
###############################################################################


def _recurse_buffer_length(batch_buffer, length, prev_keys):
    # Just to hide the extra arguments from the API.
    for k, v in batch_buffer.items():
        if isinstance(v, dict):
            length = _recurse_buffer_length(v, length, prev_keys + [k])
        elif length is None:
            length = len(v)
        elif length != len(v):
            raise RuntimeError("Different lengths in buffer: {}".format(
                prev_keys + [k]))
    return length


def _recurse_segment(batch_buffer, start, segment_length):
    segment = struct()
    for k, v in batch_buffer.items():
        if isinstance(v, dict):
            segment[k] = _recurse_segment(v, start, segment_length)
        else:
            segment[k] = v[start:start + segment_length]
    return segment


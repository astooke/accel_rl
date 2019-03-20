

class struct(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

    def copy(self):
        """
        Provides a "deep copy" of all unbroken chains of types (struct, dict,
        list), but shallow copies otherwise, i.e. no data is actually copied
        (e.g. numpy arrays are NOT copied)
        """
        new_dict = dict()
        for k, v in self.__dict__.items():
            new_dict[k] = _struct_copy(v)
        return struct(**new_dict)


def _struct_copy(obj):
    if isinstance(obj, (dict, list, struct)):
        obj = obj.copy()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list, struct)):
                obj[k] = _struct_copy(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list, struct)):
                obj[i] = _struct_copy(v)
    return obj


def nbytes_unit(nbytes):
    for unit in ["KB", "MB", "GB"]:
        nbytes /= 1024.
        if nbytes < 1000:
            break
    return nbytes, unit


def make_seed():
    """
    Returns a random number between [0, 10000], using timing jitter.

    This has a white noise spectrum and gives unique values for multiple
    simultaneous processes...some simpler attempts did not achieve that,
    although there's probably a better way.
    """
    import time
    d = 10000
    t = time.time()
    sub1 = int(t * d) % d
    sub2 = int(t * d ** 2) % d
    s = 1e-3
    s_inv = 1. / s
    time.sleep(s * sub2 / d)
    t2 = time.time()
    t2 = t2 - int(t2)
    t2 = int(t2 * d * s_inv) % d
    time.sleep(s * sub1 / d)
    t3 = time.time()
    t3 = t3 - int(t3)
    t3 = int(t3 * d * s_inv * 10) % d
    return (t3 - t2) % d

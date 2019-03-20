
from inspect import getargspec


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def save_args(values, underscore=False):
    pre = "_" if underscore else ""
    args = list()
    for Cls in values['self'].__class__.__mro__:
        if '__init__' in Cls.__dict__:
            args += getargspec(Cls.__init__).args[1:]
    for a in args:
        if a in values:
            setattr(values['self'], pre + a, values[a])


def retrieve_args(obj, bunch=True):
    args = {k.lstrip("_"): v for k, v in vars(obj).items()}
    if bunch:
        args = Bunch(args)
    return args


# class Example(object):

#     def __init__(self, title, whoa=1, yeah='3'):
#         save_args(vars())

#     def initialize(self):
#         s = retrieve_args(self)
#         # (s is now short for self, gets rid of _, any only has __init__ args)
#         return s


# class Inherit(Example):

#     def __init__(self, another, one=2, **kwargs):
#         save_args(vars())  # (can go either before or after)
#         super().__init__(**kwargs)


# args = ['self']
# for C in Three.__mro__:
#     if '__init__' in C.__dict__:
#         args += inspect.getargspec(C).args[1:]



class BaseOptimizer(object):

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def optimize(self, inputs):
        raise NotImplementedError

    @property
    def paralellism_tag(self):
        raise NotImplementedError

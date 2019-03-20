

class RLAlgorithm(object):

    def initialize(self):
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    @property
    def opt_info_keys(self):
        return []

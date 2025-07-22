import numpy as np

class ReplayBuffer:
    def __init__(self, cfg):
        self.max_size = cfg.train.replay_size
        self.ptr = 0
        self.full = False
        self.storage = {}

    def add(self,):
        ## To do: add transition into the buffer
        pass
    
    def sample(self, batch_size):
        ## To do: random sample
        pass
    
    @property
    def size(self):
        return self.max_size if self.full else self.ptr
import torch.distributions as dist

class qEpsilon(dist.Beta):

    def __init__(self):
        self.asd = 1

    def update_CAVI(self):
        pass



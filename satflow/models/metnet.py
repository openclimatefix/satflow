import torch


class MetNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

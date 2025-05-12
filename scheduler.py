import torch
from torch.optim.lr_scheduler import _LRScheduler

class CustomDecay(_LRScheduler):
    def __init__(self, optimizer, total_epochs, min_factor, last_epoch=-1):
        self.total_epochs = total_epochs
        self.min_factor = min_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        factor = torch.max(torch.Tensor([1 - epoch/self.total_epochs, self.min_factor]))
        return [base_lr * factor for base_lr in self.base_lrs]
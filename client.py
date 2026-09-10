import math

class SGDRCosineScheduler:
    """
    SGDR: Cosine Annealing with Warm Restarts.
    Loshchilov & Hutter (2017).
    """
    def __init__(self, base_lr=0.1, min_lr=1e-4, t_0=10, t_mult=2):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.t_0 = t_0
        self.t_mult = t_mult
        self.t_i = t_0
        self.t_cur = 0

    def get_lr(self, epoch):
        # Calculate current cycle progress
        cycle_len = self.t_0
        curr_epoch = epoch
        
        while curr_epoch >= cycle_len:
            curr_epoch -= cycle_len
            cycle_len *= self.t_mult

        ratio = curr_epoch / cycle_len
        lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * ratio))
        return lr

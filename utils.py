import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def kl_loss_function(z_mu, z_logvar):
    '''
    z_mu.shape: (B, z_dim)
    z_logvar.shape: (B, z_dim)
    '''
    z_var = torch.exp(z_logvar)
    return - 0.5 * (1 + z_logvar - (z_mu ** 2) - z_var).sum(dim=1)


def recon_loss_function(x, x_hat):
    return ((x - x_hat) ** 2).sum(dim=1)




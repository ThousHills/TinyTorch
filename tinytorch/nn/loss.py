import tinytorch

from .module import Module


class BCELoss(Module):

    def __init__(self, method='mean'):
        self.method = method

    def forward(self, pred, target):
        loss = -target*pred.log() - (1-target)*(1-pred).log()
        return loss.mean()

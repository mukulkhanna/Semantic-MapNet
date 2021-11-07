import torch
from torch import nn


class SemmapLoss(nn.Module):
    def __init__(self, weights=None):
        super(SemmapLoss, self).__init__()
        if weights is None:
            self.loss = nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss = nn.CrossEntropyLoss(weight=weights, reduction="none")
            print("Using weights for CE loss.")
        

    def forward(self, obj_gt, obj_pred, mask):
        mask = mask.float()
        loss = self.loss(obj_pred, obj_gt)
        loss = torch.mul(loss, mask)
        # -- mask is assumed to have a least one value
        loss = loss.sum() / mask.sum()
        return loss

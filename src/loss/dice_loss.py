import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets = F.one_hot(targets, num_classes=probs.shape[1]).permute(0,4,1,2,3).float()

        dims = (0,2,3,4)
        intersection = torch.sum(probs * targets, dims)
        union = torch.sum(probs + targets, dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

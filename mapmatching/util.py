import torch.nn as nn
import torch.nn.functional as F


class DenseLoss(nn.Module):
    def __init__(self, kind='CE'):
        super(DenseLoss, self).__init__()
        self.kind = kind
        if kind == "KL":
            self.loss = F.kl_div()
        elif kind == "CE":
            self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y):
        target, mask = y
        target = target[1:]
        mask = mask[1:]
        if self.kind == "KL":
            y_pred = F.log_softmax(y_pred, -1)
            target = F.softmax(target, -1)
            loss = self.loss(
                y_pred[mask == 1],
                target[mask == 1]
            )
            return loss
        else:
            loss = self.loss(
                y_pred[mask == 1],
                target[mask == 1]
            )
            return loss

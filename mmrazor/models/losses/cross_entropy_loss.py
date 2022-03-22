# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(self, loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        preds_T = preds_T.detach()
        loss = F.cross_entropy(preds_S, preds_T.argmax(dim=1))
        return loss * self.loss_weight

# Copyright (c) OpenMMLab. All rights reserved.
from .cwd import ChannelWiseDivergence
from .kl_divergence import KLDivergence
from .weighted_soft_label_distillation import WSLD
from .cross_entropy_loss import CrossEntropyLoss

__all__ = ['ChannelWiseDivergence', 'KLDivergence', 'WSLD', 'CrossEntropyLoss']

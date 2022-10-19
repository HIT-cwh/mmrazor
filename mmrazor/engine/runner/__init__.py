# Copyright (c) OpenMMLab. All rights reserved.
from .darts_loop import DartsEpochBasedTrainLoop, DartsIterBasedTrainLoop
from .distill_val_loop import SelfDistillValLoop, SingleTeacherDistillValLoop
from .evolution_search_loop import EvolutionSearchLoop
from .iteprune_val_loop import ItePruneValLoop
from .greedy_search_loop import AutoSlimGreedySearchLoop
from .slimmable_val_loop import SlimmableValLoop
from .subnet_sampler_loop import GreedySamplerTrainLoop
from .subnet_val_loop import SubnetValLoop

__all__ = [
    'SingleTeacherDistillValLoop', 'DartsEpochBasedTrainLoop',
    'DartsIterBasedTrainLoop', 'SlimmableValLoop', 'EvolutionSearchLoop',
    'GreedySamplerTrainLoop', 'SubnetValLoop', 'SelfDistillValLoop',
    'ItePruneValLoop', 'AutoSlimGreedySearchLoop'
]

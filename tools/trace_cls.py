# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import torch
from mmcls.structures import ClsDataSample
from mmengine.config import Config, DictAction

from mmrazor.models.utils import CustomTracer
from mmrazor.registry import MODELS
from mmrazor.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Train an algorithm')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    register_all_modules(False)
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # build model
    model = MODELS.build(cfg.model)
    tracer = CustomTracer()
    model.ori_forward = model.forward
    # model.forward = model.extract_feat
    model.forward = model.loss
    dummy_input = torch.rand(1, 3, 224, 224)
    data_samples = [ClsDataSample().set_gt_label(1)]
    import pdb
    pdb.set_trace()
    traced = tracer.trace(
        model,
        concrete_args={
            'inputs': dummy_input,
            'data_samples': data_samples
        })
    print(traced)


if __name__ == '__main__':
    main()

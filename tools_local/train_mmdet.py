# -*- coding: utf-8 -*-
"""mmdet 通用训练启动脚本（替代 .mim/tools/train.py，带 PyTorch 2.6+ 补丁）。"""
import argparse

import torch

_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--work-dir', required=True)
    args = ap.parse_args()

    import mmdet  # noqa: F401 触发注册
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()

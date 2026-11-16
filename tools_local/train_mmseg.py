# -*- coding: utf-8 -*-
"""mmseg/mmengine 通用训练启动脚本（替代 .mim/tools/train.py）。

含 PyTorch 2.6+ 兼容补丁：torch.load 默认 weights_only=True，
mmengine 0.10.7 加载含 HistoryBuffer 的 checkpoint 会 UnpicklingError，
对本地可信 checkpoint 统一回退 weights_only=False。
"""
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

    import mmseg  # noqa: F401 触发注册
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()

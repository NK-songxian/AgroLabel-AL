# -*- coding: utf-8 -*-
"""mmseg/mmengine 通用训练启动脚本（替代 .mim/tools/train.py）。"""
import argparse

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

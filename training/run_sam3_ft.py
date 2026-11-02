# -*- coding: utf-8 -*-
"""
run_sam3_ft.py —— SAM3 官方训练管线的"一键启动器"（F1 / I-01 / I-02 共用）

三件事，顺序做，任一步失败就立刻退出（不要带着半套配置去训练）：

  1. 用 datasets_tools/sam3_make_ft_config.py 生成 Hydra 配置；
  2. 把配置拷进 `third_party/sam3_repo/sam3/train/configs/plantseg34/`
     —— `train.py` 里 `initialize_config_module("sam3.train")` 是按模块目录解析
     `-c` 的，配置放在别处会报 "Could not find config"；
  3. 用 **sam3 训练虚拟环境**的解释器起 `sam3/train/train.py`。

为什么不用当前解释器
------------------
SAM3 训练要额外的依赖（hydra-core / fvcore / iopath / submitit 等），装在同一环境里
容易和推理侧（ultralytics 8.4）打架。仓库里既定做法是独立 venv：
`/root/autodl-tmp/Ryan/envs/sam3train/bin/python`。这里默认就用它，可用
`--train-py` 或环境变量 `SAM3_TRAIN_PY` 覆盖。

F1 / I-01 / I-02 的差别只有参数，不是三份代码：
  F1     --lora-rank 0                        全参数微调（对照）
  I-01   --lora-rank 16 --lora-targets transformer,decoder --lora-extra-trainable mask_decoder
  I-02   --lora-rank 16 同上 + --level disease / 多提示（提示鲁棒）

用法
----
  # 只生成配置、打印将要执行的命令（不训练）
  python training/run_sam3_ft.py --exp-name s2_full_ft_host --dry-run

  # F1 真跑
  python training/run_sam3_ft.py --exp-name s2_full_ft_host --max-epochs 10

  # I-01 LoRA
  python training/run_sam3_ft.py --exp-name i01_lora_r16 --lora-rank 16 \
      --lora-alpha 32 --lora-targets transformer,decoder \
      --lora-extra-trainable mask_decoder --lr-scale 1.0 --max-epochs 10
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAM3REPO = ROOT / "third_party" / "sam3_repo"
CFG_DIR = SAM3REPO / "sam3" / "train" / "configs" / "plantseg34"
DEFAULT_TRAIN_PY = "/root/autodl-tmp/Ryan/envs/sam3train/bin/python"


def parse_args():
    ap = argparse.ArgumentParser(description="SAM3 微调/ LoRA 启动器")
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--level", default="host", choices=["host", "disease"])
    ap.add_argument("--phrase-index", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--val-freq", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=1008)
    ap.add_argument("--chunk", type=int, default=34)
    ap.add_argument("--train-workers", type=int, default=8)
    ap.add_argument("--lr-scale", type=float, default=0.1)
    ap.add_argument("--limit-ids", type=int, default=0, help="训练图数上限（冒烟）")
    ap.add_argument("--val-images", type=int, default=100)
    ap.add_argument("--train-neg", default="true", choices=["true", "false"])
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-targets", default="")
    ap.add_argument("--lora-extra-trainable", default="")
    ap.add_argument("--train-py", default=os.environ.get("SAM3_TRAIN_PY", DEFAULT_TRAIN_PY))
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    a = parse_args()

    if not SAM3REPO.exists():
        raise SystemExit(f"找不到 SAM3 源码目录：{SAM3REPO}（F1/I-01/I-02 需要它）")

    cfg_name = f"{a.exp_name}.yaml"
    tmp_cfg = ROOT / "configs" / "sam3" / cfg_name

    gen = [
        sys.executable, str(ROOT / "datasets_tools" / "sam3_make_ft_config.py"),
        "--exp-name", a.exp_name,
        "--out", str(tmp_cfg),
        "--level", a.level,
        "--phrase-index", str(a.phrase_index),
        "--max-epochs", str(a.max_epochs),
        "--val-freq", str(a.val_freq),
        "--resolution", str(a.resolution),
        "--chunk", str(a.chunk),
        "--train-workers", str(a.train_workers),
        "--lr-scale", str(a.lr_scale),
        "--val-images", str(a.val_images),
        "--train-neg", a.train_neg,
        "--logdir", str(ROOT / "runtime" / "runs"),
    ]
    if a.limit_ids:
        gen += ["--limit-ids", str(a.limit_ids)]
    if a.lora_rank > 0:
        gen += ["--lora-rank", str(a.lora_rank), "--lora-alpha", str(a.lora_alpha),
                "--lora-dropout", str(a.lora_dropout),
                "--lora-targets", a.lora_targets,
                "--lora-extra-trainable", a.lora_extra_trainable]

    print(f"[1/3] 生成配置: {' '.join(gen)}", flush=True)
    rc = subprocess.call(gen, cwd=str(ROOT))
    if rc != 0:
        raise SystemExit(f"配置生成失败 rc={rc}")

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    dst = CFG_DIR / cfg_name
    shutil.copyfile(tmp_cfg, dst)
    print(f"[2/3] 配置已放入 {dst}", flush=True)

    rel_cfg = f"configs/plantseg34/{cfg_name}"
    cmd = [a.train_py, "sam3/train/train.py", "-c", rel_cfg,
           "--use-cluster", "0", "--num-gpus", str(a.num_gpus)]
    env = {**os.environ}
    # Hydra 在 fork 出来的进程里也要 import 到 src.lora_builder，所以仓库根必须上 PYTHONPATH
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.setdefault("YOLO_OFFLINE", "1")
    print(f"[3/3] cd {SAM3REPO} && {' '.join(cmd)}", flush=True)
    print(f"      PYTHONPATH={env['PYTHONPATH']}", flush=True)

    if a.dry_run:
        print("[dry-run] 不执行训练")
        return
    if not Path(a.train_py).exists():
        raise SystemExit(f"训练解释器不存在：{a.train_py}（用 --train-py 指定）")

    rc = subprocess.call(cmd, cwd=str(SAM3REPO), env=env)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

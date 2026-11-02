#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F/I 训练/评测队列执行器。

按 `experiments/queue.yaml` 顺序执行，默认先等 GPU 上现有训练（B 系列）跑完再开始，
避免并发抢卡 OOM。每个任务的日志写到 runtime/logs/queue/<id>_<ts>.log，
运行状态写 experiments/queue_state.json。

用法：
  python experiments/run_queue.py --list
  python experiments/run_queue.py --wait-gpu            # 等 B 系列跑完再顺序跑全部
  python experiments/run_queue.py --only F2             # 只跑 F2
  python experiments/run_queue.py --no-wait             # 不等，直接跑（谨慎）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "experiments" / "queue.yaml"
STATE = ROOT / "experiments" / "queue_state.json"
LOG_DIR = ROOT / "runtime" / "logs" / "queue"

# 判断"GPU 忙"的脚本特征（B 系列 + F/I 训练与 GPU 评测）
TRAIN_SIGNS = (
    "train_seg_baseline.py",
    "train_yoloe_seg.py",
    "train_yoloworld.py",
    "run_sam3_ft.py",          # SAM3 微调启动器（内部再起 sam3/train/train.py）
    "sam3/train/train.py",
    "m13_box_prompt_eval.py",   # 以下三个是 GPU 评测，占卡也要等
    "i03_ondemand_refine.py",
    "i04_decoupled_eval.py",
)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_state(eid: str, entry: dict) -> None:
    cur = {}
    if STATE.exists():
        try:
            cur = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    cur[eid] = entry
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def busy_procs() -> list[str]:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    hits = []
    for line in out.splitlines():
        if "run_queue" in line:
            continue
        if any(s in line for s in TRAIN_SIGNS):
            hits.append(line.strip())
    return hits


def resolve(cmd: str) -> str:
    return cmd.replace("{py}", sys.executable).replace("{repo}", str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="F/I 队列执行器")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="只跑指定 id")
    ap.add_argument("--wait-gpu", action="store_true", help="等现有训练结束再开始（默认开）")
    ap.add_argument("--no-wait", action="store_true", help="不等 GPU，直接跑")
    ap.add_argument("--poll", type=int, default=60, help="等待 GPU 的轮询秒数")
    a = ap.parse_args()

    q = load_yaml(QUEUE)
    items = q.get("queue", [])
    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    if a.list or not items:
        print(f"{'ID':<6}{'状态':<9}名称")
        print("-" * 60)
        for it in items:
            st = state.get(it["id"], {}).get("status", it.get("status", "pending"))
            print(f"{it['id']:<6}{st:<9}{it.get('name','')}")
        return

    wait = not a.no_wait
    for it in items:
        eid = it["id"]
        if a.only and eid != a.only:
            continue
        cmd = it.get("command")
        if not cmd:
            print(f"[{eid}] 无 command，跳过（{it.get('note','')}）")
            continue

        if wait:
            hits = busy_procs()
            while hits:
                print(f"[{eid}] GPU 忙，等待中：{hits[0][:90]} ...", flush=True)
                time.sleep(a.poll)
                hits = busy_procs()

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = LOG_DIR / f"{eid}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        real = resolve(cmd)
        print(f"[{eid}] 运行: {real}\n        日志: {log}", flush=True)
        t0 = time.time()
        env = {**os.environ}
        env.setdefault("YOLO_OFFLINE", "1")
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.call(real, shell=True, cwd=str(ROOT), env=env,
                                 stdout=f, stderr=subprocess.STDOUT)
        entry = {
            "status": "done" if rc == 0 else "failed",
            "rc": rc,
            "sec": round(time.time() - t0, 1),
            "log": str(log),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_state(eid, entry)
        print(f"[{eid}] {'完成' if rc == 0 else '失败'} rc={rc} ({entry['sec']}s)", flush=True)


if __name__ == "__main__":
    main()

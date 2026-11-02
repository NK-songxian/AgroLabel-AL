#!/usr/bin/env python
"""实验矩阵执行器：按 `experiments/matrix.yaml` 统一调度 B/Z/F/I 各实验。

用法::

    python experiments/run_matrix.py --list
    python experiments/run_matrix.py --dry-run B0
    python experiments/run_matrix.py --run B0
    python experiments/run_matrix.py --phase baseline
    python experiments/run_matrix.py --run B0 --allow-busy   # 训练在跑也强制执行

设计:
- 只读 matrix.yaml；运行状态写 experiments/matrix_state.json，不改 yaml。
- 默认检测到 train.py / prompt_sweep 进程时**拒绝执行**（避免抢卡影响在跑训练），
  `--allow-busy` 放行。
- 命令占位符 {repo} {data_root} {data_yaml} {weights} {sam3_weights} {out}
  由 meta 与 --data-root 解析；云端/本地自动选数据根。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "experiments" / "matrix.yaml"
STATE = ROOT / "experiments" / "matrix_state.json"
LOG_DIR = ROOT / "runtime" / "logs" / "matrix"


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def busy_procs() -> list[str]:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    hits = []
    for line in out.splitlines():
        if "run_matrix" in line:
            continue
        if "train.py" in line or "prompt_sweep" in line:
            hits.append(line.strip())
    return hits


def resolve(tpl: str, meta: dict, out_dir: Path, data_root_override: str | None) -> str:
    cloud = meta.get("data_root_cloud", "")
    local = meta.get("data_root_local", "")
    if data_root_override:
        data_root = data_root_override
    elif cloud and Path(cloud).exists():
        data_root = cloud
    else:
        data_root = local
    return (
        tpl.replace("{repo}", str(ROOT))
        .replace("{data_root}", str(data_root))
        .replace("{data_yaml}", str(Path(data_root) / "data.yaml"))
        .replace("{sam3_weights}", str(meta.get("sam3_weights", "")))
        .replace("{weights}", str(meta.get("weights", "")))
        .replace("{out}", str(out_dir))
    )


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="实验矩阵执行器")
    ap.add_argument("--list", action="store_true", help="列出全部实验")
    ap.add_argument("--dry-run", metavar="ID", help="只打印最终命令")
    ap.add_argument("--run", metavar="ID", help="运行单个实验")
    ap.add_argument("--phase", help="顺序运行某阶段：baseline/zeroshot/finetune/innovation")
    ap.add_argument("--data-root", default=None, help="覆盖数据根目录")
    ap.add_argument("--allow-busy", action="store_true", help="有训练在跑也执行")
    args = ap.parse_args()

    m = load_yaml(MATRIX)
    meta = m.get("meta", {})
    exps = {e["id"]: e for e in m["experiments"]}
    state = load_state()

    def status_of(eid: str, default: str) -> str:
        return state.get(eid, {}).get("status", default)

    def print_list() -> None:
        print(f"{'ID':<6}{'阶段':<11}{'优先级':<12}{'状态':<9}名称")
        print("-" * 78)
        for e in m["experiments"]:
            print(
                f"{e['id']:<6}{e.get('phase',''):<11}{e.get('priority',''):<12}"
                f"{status_of(e['id'], e.get('status','todo')):<9}{e['name']}"
            )

    def run_one(e: dict, allow_busy: bool, dry: bool = False) -> bool:
        eid = e["id"]
        tpl = e.get("command")
        if not tpl:
            print(f"[{eid}] 无 command（status={e.get('status')}, needs={e.get('needs')}）")
            return False
        out_dir = ROOT / "experiments" / "matrix" / eid
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = resolve(tpl, meta, out_dir, args.data_root)

        if dry:
            print(cmd)
            return True

        if not allow_busy:
            hits = busy_procs()
            if hits:
                print("[拒绝] 检测到正在运行的任务，加 --allow-busy 可强制：")
                for h in hits[:3]:
                    print("  " + h)
                return False

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = LOG_DIR / f"{eid}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        print(f"[{eid}] 运行: {cmd}\n        日志: {log}")
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.call(cmd, shell=True, cwd=str(ROOT),
                                 stdout=f, stderr=subprocess.STDOUT)
        state[eid] = {
            "status": "done" if rc == 0 else "failed",
            "rc": rc,
            "sec": round(time.time() - t0, 1),
            "log": str(log),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_state(state)
        print(f"[{eid}] {'完成' if rc == 0 else '失败'} rc={rc} ({state[eid]['sec']}s)")
        return rc == 0

    if args.list or (not args.run and not args.phase and not args.dry_run):
        print_list()
        return

    if args.dry_run:
        e = exps.get(args.dry_run)
        if not e:
            sys.exit(f"未知实验 ID：{args.dry_run}")
        run_one(e, allow_busy=True, dry=True)
        return

    if args.run:
        e = exps.get(args.run)
        if not e:
            sys.exit(f"未知实验 ID：{args.run}")
        sys.exit(0 if run_one(e, args.allow_busy) else 1)

    if args.phase:
        todo = [e for e in m["experiments"] if e.get("phase") == args.phase]
        if not todo:
            sys.exit(f"没有阶段 {args.phase}")
        for e in todo:
            if not e.get("command"):
                print(f"[{e['id']}] 跳过（无 command）")
                continue
            if not run_one(e, args.allow_busy):
                print(f"[{args.phase}] 中断于 {e['id']}")
                sys.exit(1)
        print(f"[{args.phase}] 全部完成")


if __name__ == "__main__":
    main()

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
- 命令占位符 {py} {repo} {data_root} {data_yaml} {weights} {sam3_weights} {out}
  由 meta 与 --data-root 解析；云端/本地自动选数据根。
- {py} 取当前解释器（sys.executable），不依赖 PATH 里的 python。
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


def pick_data_root(meta: dict, override: str | None) -> str:
    """数据根优先用仓库内 data/plantseg34（本地/云端一致），再回落云端/本地路径。"""
    inrepo = ROOT / "data" / "plantseg34"
    if override:
        return override
    if inrepo.is_dir():
        return str(inrepo)
    cloud = str(meta.get("data_root_cloud", "")).replace("{repo}", str(ROOT))
    if cloud and Path(cloud).is_dir():
        return cloud
    return meta.get("data_root_local", "")


def ensure_data_yaml(data_root: str) -> None:
    """把 data.yaml 的 path 校正为当前机器的绝对路径。

    数据目录不进 git，data.yaml 却可能带着另一台机器的路径（如 F:/_plantseg/...），
    YOLO 读到不存在的 path 会静默去找默认数据集目录。此处就地改成实际路径。
    """
    p = Path(data_root) / "data.yaml"
    if not p.is_file():
        return
    try:
        import yaml

        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return
    want = str(Path(data_root).resolve())
    if d.get("path") == want:
        return
    d["path"] = want
    # 原子写：并发启动多个实验时，避免两个进程同时写坏 data.yaml
    tmp = p.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    tmp.replace(p)
    print(f"[data.yaml] path -> {want}")


def resolve(tpl: str, meta: dict, out_dir: Path, data_root: str) -> str:
    """解析命令占位符。"""
    sam3_weights = str(meta.get("sam3_weights", "")).replace("{repo}", str(ROOT))
    return (
        tpl.replace("{repo}", str(ROOT))
        .replace("{py}", sys.executable)
        .replace("{data_root}", str(data_root))
        .replace("{data_yaml}", str(Path(data_root) / "data.yaml"))
        .replace("{sam3_weights}", sam3_weights)
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


def save_state(eid: str, entry: dict) -> None:
    """读-改-写 + 原子替换。

    并行跑多个实验时各自记住的是启动时的快照，直接整份覆盖会把别人刚写的结果冲掉；
    这里每次写前重新读一次，只合并自己这一条。
    """
    cur = load_state()
    cur[eid] = entry
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE)


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
        data_root = pick_data_root(meta, args.data_root)
        cmd = resolve(tpl, meta, out_dir, data_root)

        if dry:
            print(cmd)
            return True

        ensure_data_yaml(data_root)

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
        env = {**os.environ}
        env.setdefault("YOLO_OFFLINE", "1")  # 云端无外网，关闭 ultralytics 联网自检
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.call(cmd, shell=True, cwd=str(ROOT), env=env,
                                 stdout=f, stderr=subprocess.STDOUT)
        entry = {
            "status": "done" if rc == 0 else "failed",
            "rc": rc,
            "sec": round(time.time() - t0, 1),
            "log": str(log),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_state(eid, entry)
        print(f"[{eid}] {'完成' if rc == 0 else '失败'} rc={rc} ({entry['sec']}s)")
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

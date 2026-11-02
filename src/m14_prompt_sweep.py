# -*- coding: utf-8 -*-
"""
m14_prompt_sweep.py —— 开放词表提示词工程扫描（Z2/Z3 的提示消融）

## 为什么要有这个脚本

已有实证（experiments/sam3_s1/sweep.json，846 图，同一 SAM3 权重，只换提示）：

    host_0          "disease lesions on apple leaves"    mAP50 0.1709  dice 0.8274  n_pred 53702
    disease_0       "apple scab lesions on apple leaves" mAP50 0.1417  dice 0.8431  n_pred 39601
    disease_1       "apple scab lesions on apple"        mAP50 0.1308  dice 0.8424  n_pred 27367
    species_tuned   调优后的物种名                        mAP50 0.0376  dice 0.8538  n_pred 31184

两条结论决定了本脚本的设计：

1. **量级由"有没有病征词"决定**：提示里出现 lesion 就是 0.13~0.17，只有物种名就掉到
   0.004~0.04（与 Z2/Z3 同量级）。所以第一维度是"病征词族"，不是措辞微调。
2. **越具体不等于越好**：病害级提示 dice 更高（边界更准）但 n_pred 更少（召回降），
   AP 反而低。所以必须**同时记录 mAP50 与 dice、mAP50-95**，否则会误判成"提示退化"。

本脚本把提示拆成三个可分别控制的维度，每次只动一个，跑同一套 val：

    A 病征词族   lesion / spot / abnormal region / plain species name
    B 句法        "X on {host} leaves" / "{host} leaf with X" / "a photo of X"
    C 粒度       宿主级（host）/ 病害级（disease，用该宿主首个病名）

## 成本

模型只加载一次，每个模板族重复 `set_classes` + `val`。640 尺寸、846 图约 1–2 min/族，
全部 7 族约 10–15 分钟。冒烟用 --max-n 60。

## 用法

  python src/m14_prompt_sweep.py --kind world --weights weights/yolov8x-worldv2.pt \
      --data data/plantseg34/data.yaml --out runtime/logs/matrix/Z2sweep

  python src/m14_prompt_sweep.py --kind yoloe --weights weights/yoloe-11l-seg.pt \
      --data data/plantseg34/data.yaml --families symptom_on_leaf,disease_level --max-n 60 \
      --out /tmp/smoke
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault("YOLO_OFFLINE", "1")

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from m12_openvocab_eval import make_subset_yaml, names_from_yaml  # noqa: E402

# ---- 模板族：每个函数拿 (宿主名, 该宿主首个病名, bank 里第 k 条宿主级短语) 生成一条提示 ----
# 名字里带维度信息，方便报告里直接引用。
TEMPLATES = {
    # 基线锚点：纯物种名（= Z2/Z3 原口径），必须留着，否则无法量化"病征词"的贡献
    "species_only":      lambda h, d, ph: h,
    # A 病征词族的三种典型写法（宿主级粒度）
    "symptom_on_leaf":   lambda h, d, ph: f"disease lesions on {h} leaves",
    "symptom_spot":      lambda h, d, ph: f"disease spots on {h} leaves",
    "symptom_abnormal":  lambda h, d, ph: f"abnormal diseased regions on {h} leaves",
    # B 句法：把宿主放前面、病征放后面
    "host_with_symptom": lambda h, d, ph: f"{h} leaf with disease lesions",
    # B 句法：CLIP 对 "a photo of" 前缀敏感，单独测一条
    "photo_prefix":      lambda h, d, ph: f"a photo of disease lesions on {h} leaves",
    # C 粒度：病害级（更具体，预期 dice↑ / 召回↓）
    "disease_level":     lambda h, d, ph: f"{d} lesions on {h} leaves",
}

DEFAULT_FAMILIES = list(TEMPLATES)


def parse_args():
    ap = argparse.ArgumentParser(description="开放词表提示词扫描")
    ap.add_argument("--kind", choices=["world", "yoloe"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--host-bank", default="configs/sam3/host_prompt_bank.json")
    ap.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                    help=f"逗号分隔，可选：{','.join(TEMPLATES)}")
    ap.add_argument("--max-n", type=int, default=0, help="只跑前 N 张，冒烟用")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def build_prompts(bank_path: str, names: list[str], family: str) -> list[str]:
    """按 class_id 顺序拼出 n_classes 条提示。bank 的 class_id 是 0 基，与 names 对齐。"""
    bank = json.loads(Path(bank_path).read_text(encoding="utf-8"))
    fn = TEMPLATES[family]
    out = [""] * len(names)
    for h in bank["hosts"]:
        cid = int(h["class_id"])
        if not (0 <= cid < len(names)):
            continue
        host = str(h["host"]).lower()
        phrases = h.get("host_level_phrases") or [host]
        dis = (h.get("diseases") or [{}])[0].get("name") or "disease"
        out[cid] = fn(host, str(dis).lower(), phrases[0])
    missing = [i for i, v in enumerate(out) if not v]
    if missing:
        raise SystemExit(f"提示词库缺类别索引 {missing}")
    return out


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    from ultralytics import YOLOE, YOLOWorld

    dc = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
    names = names_from_yaml(dc)
    fams = [f.strip() for f in a.families.split(",") if f.strip()]
    bad = [f for f in fams if f not in TEMPLATES]
    if bad:
        raise SystemExit(f"未知模板族 {bad}，可选 {list(TEMPLATES)}")

    data = a.data
    if a.max_n > 0:
        data = make_subset_yaml(a.data, a.split, a.max_n)

    # 模型只加载一次；每次 set_classes 重建文本嵌入即可
    m = YOLOWorld(a.weights) if a.kind == "world" else YOLOE(a.weights)

    table = []
    for fam in fams:
        prompts = build_prompts(a.host_bank, names, fam)
        t0 = time.time()
        try:
            if a.kind == "world":
                m.set_classes(prompts)
            else:
                m.set_classes(prompts, m.get_text_pe(prompts))
            r = m.val(data=data, split=a.split, imgsz=a.imgsz, device=a.device, verbose=False)
        except Exception as e:  # noqa: BLE001
            print(f"[{fam}] 失败: {type(e).__name__}: {e}", flush=True)
            table.append({"family": fam, "error": f"{type(e).__name__}: {e}"})
            continue

        row = {"family": fam, "prompt_example": prompts[0],
               "box_map50": round(float(r.box.map50), 5),
               "box_map": round(float(r.box.map), 5),
               "box_map75": round(float(r.box.map75), 5),
               "sec": round(time.time() - t0, 1)}
        seg = getattr(r, "seg", None)
        if seg is not None:
            row.update(seg_map50=round(float(seg.map50), 5),
                       seg_map=round(float(seg.map), 5),
                       seg_map75=round(float(seg.map75), 5))
        table.append(row)
        print(f"{fam:<18} box50={row['box_map50']:.5f} "
              f"seg50={row.get('seg_map50', 0):.5f} ({row['sec']}s) | {prompts[0][:52]}",
              flush=True)

    rep = {"kind": a.kind, "weights": a.weights, "split": a.split,
           "imgsz": a.imgsz, "images": a.max_n or "all",
           "n_classes": len(names), "table": table}
    json.dump(rep, open(os.path.join(a.out, "prompt_sweep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    ok = [r for r in table if "error" not in r]
    if ok:
        metric = "seg_map50" if "seg_map50" in ok[0] else "box_map50"
        best = max(ok, key=lambda r: r.get(metric, 0))
        base = next((r for r in ok if r["family"] == "species_only"), None)
        print(f"\n最佳（{metric}）: {best['family']} = {best.get(metric)}")
        if base and base.get(metric):
            print(f"相对纯物种名基线（{base.get(metric)}）提升 "
                  f"{best.get(metric) / base.get(metric):.1f}×")
    print(f"DONE -> {a.out}/prompt_sweep.json")


if __name__ == "__main__":
    main()

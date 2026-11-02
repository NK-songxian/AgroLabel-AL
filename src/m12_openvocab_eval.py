# -*- coding: utf-8 -*-
"""
m12_openvocab_eval.py —— 开放词表零样本评测（矩阵 Z2 / Z3 用）

Z2: YOLO-World 零样本检测，只出框 AP（不与掩码 AP 混比）
Z3: YOLOE-Seg 零样本分割，出掩码 AP

Z2b/Z3b（提示对照）：同上两个模型，但把提示从"裸宿主名"换成 host_prompt_bank
里的病征短语（--prompt-source phrases）。34 类的名字是**宿主物种**（Apple/Banana），
拿它当提示，CLIP 编码到的是"苹果这个物体"，而 GT 是"叶片上的病斑"——语义错配。
这一组对照就是用来把 Z2/Z3 的低分**归因到提示写法**上，而不是模型能力。

离线环境：云端连不上 GitHub，权重由本地下载后放入 weights/；
脚本内先禁用 SSL 校验并置 YOLO_OFFLINE，避免 ultralytics 启动自检联网卡死。

用法：
  python src/m12_openvocab_eval.py --kind world \
      --weights weights/yolov8x-worldv2.pt --data data/plantseg34/data.yaml \
      --split val --device 0 --out runtime/logs/matrix/Z2.json
  python src/m12_openvocab_eval.py --kind yoloe \
      --weights weights/yoloe-11l-seg.pt --data data/plantseg34/data.yaml \
      --split val --device 0 --out runtime/logs/matrix/Z3.json
  冒烟（只跑 N 张）：加 --max-n 5
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import tempfile
from pathlib import Path

# ---- 必须在 import ultralytics 之前：离线兜底 ----
ssl._create_default_https_context = ssl._create_unverified_context
_orig_create = ssl.create_default_context


def _patched_create(*a, **kw):
    try:
        return _orig_create(*a, **kw)
    except Exception:
        return ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


ssl.create_default_context = _patched_create
os.environ.setdefault("YOLO_OFFLINE", "1")

import yaml  # noqa: E402
from ultralytics import YOLOE, YOLOWorld  # noqa: E402

# ---- 修补 YOLOWorld.set_classes 的 CLIP 设备错位（GPU 上才会暴露）----
try:
    from ultralytics.nn.text_model import CLIP as _CLIP

    _orig_encode_text = _CLIP.encode_text

    def _encode_text_dev(self, texts, dtype=None):
        try:
            dev = next(self.model.parameters()).device
            if hasattr(texts, "to"):
                texts = texts.to(dev)
        except Exception:
            pass
        return _orig_encode_text(self, texts, dtype=dtype)

    _CLIP.encode_text = _encode_text_dev
except Exception as e:  # pragma: no cover
    print(f"[warn] CLIP 补丁未生效: {e}")


def names_from_yaml(d: dict) -> list[str]:
    n = d.get("names")
    if isinstance(n, dict):
        return [n[i] for i in sorted(n)]
    return list(n)


def prompts_from_bank(bank_path: str, n_classes: int, phrase_index: int = 0) -> list[str]:
    """用 host_prompt_bank 的寄主级短语替换裸类名，作为开放词表提示。

    为什么需要这一步：34 类的名字是**宿主物种**（Apple / Banana …），
    拿它当文本提示，CLIP 编码出来的是"苹果这个物体"，而 GT 是"叶片上的病斑"，
    语义对不上。host_prompt_bank 里的短语（"disease lesions on apple leaves"）
    描述的是病斑本身，才是这张图里要分割的东西。

    bank 的 class_id 是 0 基，与 data.yaml 的 names 下标一一对齐。
    """
    bank = json.loads(Path(bank_path).read_text(encoding="utf-8"))
    out: list[str] = [""] * n_classes
    for h in bank["hosts"]:
        cid = int(h["class_id"])
        if not (0 <= cid < n_classes):
            continue
        ph = h.get("host_level_phrases") or []
        if ph:
            out[cid] = ph[min(phrase_index, len(ph) - 1)]
    missing = [i for i, v in enumerate(out) if not v]
    if missing:
        raise SystemExit(f"提示词库缺这些类别索引: {missing}（共 {n_classes} 类）")
    return out


def make_subset_yaml(data_yaml: str, split: str, n: int) -> str:
    """把 split 截成前 n 张，生成临时 data.yaml（val 指向图片清单）。"""
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(d.get("path") or Path(data_yaml).parent)
    rel = d.get(split)
    p = Path(rel)
    p = p if p.is_absolute() else root / p
    imgs = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg")))[:n]
    if not imgs:
        raise SystemExit(f"{p} 下没有图片")
    tmpdir = Path(tempfile.mkdtemp(prefix="zsub_"))
    lst = tmpdir / f"{split}.txt"
    lst.write_text("\n".join(str(x) for x in imgs), encoding="utf-8")
    nd = dict(d)
    nd[split] = str(lst)
    ny = tmpdir / "data_subset.yaml"
    ny.write_text(yaml.safe_dump(nd, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[subset] {len(imgs)} 张 -> {ny}")
    return str(ny)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["world", "yoloe"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-n", type=int, default=0, help="只跑前 N 张，冒烟用")
    ap.add_argument("--prompt-source", choices=["names", "phrases"], default="names",
                    help="names=裸宿主名（Z2/Z3 原口径）；phrases=病征短语（Z2b/Z3b 提示对照）")
    ap.add_argument("--host-bank", default="configs/sam3/host_prompt_bank.json")
    ap.add_argument("--phrase-index", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dc = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
    names = names_from_yaml(dc)
    prompts = (prompts_from_bank(a.host_bank, len(names), a.phrase_index)
               if a.prompt_source == "phrases" else names)
    print(f"[{a.kind}] classes={len(names)} prompt_source={a.prompt_source} "
          f"first3={prompts[:3]} device={a.device}")

    data = a.data
    if a.max_n > 0:
        data = make_subset_yaml(a.data, a.split, a.max_n)

    if a.kind == "world":
        m = YOLOWorld(a.weights)
        m.set_classes(prompts)
    else:
        m = YOLOE(a.weights)
        m.set_classes(prompts, m.get_text_pe(prompts))

    r = m.val(data=data, split=a.split, imgsz=a.imgsz, device=a.device, verbose=False)

    res = {
        "kind": a.kind,
        "weights": a.weights,
        "split": a.split,
        "imgsz": a.imgsz,
        "n_classes": len(names),
        "prompt_source": a.prompt_source,
        "prompts_used": prompts[:5],
        "box_map50": float(r.box.map50),
        "box_map": float(r.box.map),
        "box_map75": float(r.box.map75),
    }
    seg = getattr(r, "seg", None)
    if seg is not None:
        res.update(
            seg_map50=float(seg.map50),
            seg_map=float(seg.map),
            seg_map75=float(seg.map75),
        )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""从仓库自带的评测可视化图里裁出可标注的样例图。

背景：`experiments/sam3_conversion_preview/*.jpg` 是「原图 | 标注叠加」的双联对比图，
左半是干净的原图（无占位验证）。把它们裁出来放进 `out/demo/`，前端的 DEMO 数据和
后端的研发联调就有一批**真实的病斑图**可用，不必依赖还没到位的 F:/_plantseg。

用法（仓库根目录）：
    .\\.venv\\Scripts\\python.exe backend/tools/build_demo_set.py

产出（整目录都在 .gitignore 里，不会提交）：
    out/demo/<原名>.jpg        裁好的原图
    out/demo/_manifest.json    宽高 / host / disease / class_id
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "sam3_conversion_preview"
OUT = ROOT / "out" / "demo"

# 文件名形如 16_apple_scab_google_0111.jpg / 15_wheat_bacterial_leaf_streak_(black_chaff)_Bing_0003.jpg
NAME_RE = re.compile(r"^(\d+)_(.+?)_(google|bing|baidu)_(\d+)$", re.IGNORECASE)


def load_classes() -> list[str]:
    import yaml

    cfg = ROOT / "configs" / "dataset_plantseg34.local.yaml"
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    names = {int(k): str(v) for k, v in raw["names"].items()}
    return [names[i] for i in range(int(raw["nc"]))]


def split_left_half(im: Image.Image) -> Image.Image:
    """双联图的左半。留一点余量避免把中缝的分隔线切进来。"""
    w, h = im.size
    return im.crop((0, 0, int(w * 0.5) - 2, h))


def main() -> int:
    if not SRC.is_dir():
        print(f"找不到源目录: {SRC}", file=sys.stderr)
        return 1

    classes = load_classes()
    by_lower = {c.lower(): i for i, c in enumerate(classes)}

    OUT.mkdir(parents=True, exist_ok=True)
    entries = []

    for src in sorted(SRC.glob("*.jpg")):
        m = NAME_RE.match(src.stem)
        # host 取文件名里的第一个词（apple / wheat / corn ...）
        host_word = (m.group(2).split("_")[0] if m else src.stem.split("_")[1]).lower()
        disease = m.group(2).replace("_", " ") if m else None
        class_id = by_lower.get(host_word)
        class_name = classes[class_id] if class_id is not None else None

        with Image.open(src) as im:
            im = im.convert("RGB")
            left = split_left_half(im)
            dst = OUT / src.name
            left.save(dst, quality=92, optimize=True)

        entries.append({
            "file": src.name,
            "width": left.width,
            "height": left.height,
            "class_id": class_id,
            "class_name": class_name,
            "disease": disease,
            "source": str(src.relative_to(ROOT)).replace("\\", "/"),
        })
        print(f"  {left.width}x{left.height}  {class_name or '?':<12} {src.name}")

    manifest = {
        "note": "由 backend/tools/build_demo_set.py 从 sam3_conversion_preview 左半裁出；仅供联调",
        "count": len(entries),
        "items": entries,
    }
    (OUT / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    matched = sum(1 for e in entries if e["class_id"] is not None)
    print(f"\n完成：{len(entries)} 张 -> {OUT}")
    print(f"      其中 {matched} 张匹配上 34 类，{len(entries) - matched} 张类别未知")
    return 0


if __name__ == "__main__":
    sys.exit(main())

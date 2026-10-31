"""把 PlantDoc（只有框、没有掩码）整理成 YOLO 检测格式。

它的定位和 PlantSeg 不同，不要混淆：

  PlantSeg  —— 有真值掩码，用来**训练**分割模型，也是分割精度的评估集
  PlantDoc  —— 只有叶片级外接框，且在自然光、杂乱背景下采集，
               用来做**跨域泛化测试**：模型在 PlantSeg 上训练，
               直接拿到 PlantDoc 上算框级 mAP，掉多少一目了然

正因为只做评估，所以不用重新训练，也就不需要掩码。

数据来源：HuggingFace `Project-AgML/plant_doc_detection`（AgML 农业基准项目整理），
3 个 parquet 分片，图片内嵌在 parquet 里。标注为 COCO 格式
bbox=[x, y, w, h]，categories 为 0~28 共 29 类。

用法：
  python src/plantdoc_prepare.py --limit 20    # 先试 20 张
  python src/plantdoc_prepare.py               # 全量
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

RAW = Path("H:/Plant/_raw/plantdoc")

# 类别名取自数据集 README 的 features 定义，顺序即索引，不能改
CLASSES = [
    "Corn leaf blight", "Tomato Early blight leaf", "Potato leaf early blight",
    "Potato leaf late blight", "Blueberry leaf", "grape leaf black rot",
    "Bell_pepper leaf spot", "Cherry leaf", "Peach leaf", "Soyabean leaf",
    "Strawberry leaf", "Apple Scab Leaf", "Corn rust leaf", "Apple leaf",
    "Corn Gray leaf spot", "Tomato leaf mosaic virus", "Tomato mold leaf",
    "Tomato leaf yellow virus", "Tomato leaf bacterial spot",
    "Tomato leaf late blight", "Squash Powdery mildew leaf", "Bell_pepper leaf",
    "grape leaf", "Apple rust leaf", "Tomato Septoria leaf spot", "Tomato leaf",
    "Raspberry leaf", "Potato leaf", "Tomato two spotted spider mites leaf",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(Path(args.raw).glob("*.parquet"))
    if not files:
        raise SystemExit(f"{args.raw} 下没有 parquet，下载是否完成？")

    out = PROJECT / "out" / "dataset_plantdoc"
    for sub in ["images", "labels"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    import cv2

    n_img, n_box, bad = 0, 0, Counter()
    per_class = Counter()
    budget = args.limit if args.limit else 10 ** 9

    for fp in files:
        # 下载未完成的分片会在这里报错，跳过就好，不要整体中断
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"  跳过 {fp.name}：读不了（可能还在下载）—— {type(e).__name__}")
            continue
        for _, row in df.iterrows():
            if n_img >= budget:
                break
            img = row["image"]
            data = img.get("bytes") if isinstance(img, dict) else None
            if not data:
                bad["图片为空"] += 1
                continue
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                bad["解码失败"] += 1
                continue
            h, w = bgr.shape[:2]

            obj = row["objects"]
            # pandas 会把嵌套 list 变成 ndarray，不能写 `x or []`，
            # 那会触发「数组真值歧义」异常
            boxes = obj.get("bbox") if hasattr(obj, "get") else None
            cats = obj.get("categories") if hasattr(obj, "get") else None
            boxes = [] if boxes is None else list(boxes)
            cats = [] if cats is None else list(cats)
            if len(boxes) != len(cats):
                bad["框与类别数不一致"] += 1
                continue

            lines = []
            for (x, y, bw, bh), c in zip(boxes, cats):
                c = int(c)
                if not (0 <= c < len(CLASSES)):
                    bad["类别越界"] += 1
                    continue
                # 原始框有溢出图像边界的情况（边缘叶片、裁切图），
                # 不裁掉的话归一化坐标会大于 1，训练时告警、指标也不干净
                x1 = max(0.0, float(x))
                y1 = max(0.0, float(y))
                x2 = min(float(w), float(x) + float(bw))
                y2 = min(float(h), float(y) + float(bh))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    bad["退化框"] += 1
                    continue
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                nw = (x2 - x1) / w
                nh = (y2 - y1) / h
                if not (0 < nw <= 1 and 0 < nh <= 1):
                    bad["归一化越界"] += 1
                    continue
                lines.append(f"{c} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                per_class[CLASSES[c]] += 1
            if not lines:
                bad["无有效框"] += 1
                continue

            stem = f"plantdoc_{n_img:06d}"
            path = out / "images" / (stem + ".jpg")
            # 这批图本身是 jpg，重新编码一次保证扩展名和内容一致
            cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(path))
            (out / "labels" / (stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            n_img += 1
            n_box += len(lines)

        if n_img >= budget:
            break

    # 只做评估，所以 train/val 都指向同一份，避免 ultralytics 报错
    yaml_path = out / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out.as_posix()}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("test: images\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")

    with open(out / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": CLASSES,
                   "note": "PlantDoc 仅作跨域泛化评估，训练仍用 PlantSeg"},
                  f, ensure_ascii=False, indent=2)

    print(f"转换完成：{n_img} 张图 / {n_box} 个框 / {len(CLASSES)} 类")
    for k, v in bad.most_common():
        print(f"  丢弃 {k}: {v}")
    print(f"  平均每图 {n_box/max(n_img,1):.2f} 个框（多叶场景）")
    if per_class:
        cnt = sorted(per_class.values(), reverse=True)
        print(f"  每类框数：最大 {cnt[0]}  中位 {int(np.median(cnt))}  最小 {cnt[-1]}")
    print(f"  配置：{yaml_path}")


if __name__ == "__main__":
    main()

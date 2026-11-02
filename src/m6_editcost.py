"""把「模型预测 vs 真值」直接折算成「人工要改几下」。

## 为什么要有这个脚本

比赛有两组硬指标：

  - 同精度下，单张预标注的人工修改次数下降 ≥30%
  - 同等标注质量所需标注量，较随机采样下降 ≥20%

这两条以前只能靠「请个人来点点计时」测，既不可复现，也没法做多轮消融。
但只要有带真值的测试集，编辑成本就可以**从预测与真值的差集直接算出来**，
不需要真人也能得出稳定、可复现的数字。真人实测只用来做一次核对
（见 --human-check），证明这个自动指标和人的感受方向一致。

## 一次编辑的定义

一条实例在真值里有、预测里没有 → 标注员要「新画一笔」，记 1 次。
预测里有、真值里没有 → 要「删掉」，记 1 次。
两边都有但类别错 → 要「改类别」，记 1 次。
两边都有、类别对、但掩码 IoU 偏低（0.5~0.75）→ 要「拖一下边界」，记 1 次。
IoU ≥ 0.75 且类别对 → 直接确认，记 0 次。

整张图真值里没有病斑、预测却给出实例 → 要「标为无目标」再删干净，记 1 次。

## 匹配规则

同一类内做贪心 IoU 匹配，阈值 0.5。不做跨类匹配——按 IoU 跨类配上的
其实也是「框对了但类别错」，走「改类别」这条更贴近人的实际操作。

## 用法

  # 评估某个权重在 test 上的编辑成本
  python src/m6_editcost.py --data H:/Plant/_derived/dataset_plantseg_115/data.yaml \
      --weights runs/plantseg115/weights/best.pt --split test --name plantseg115

  # 多组权重横向对比（主动学习消融用）
  python src/m6_editcost.py --data ... --split test \
      --weights runs/al_r1/weights/best.pt runs/al_r2/weights/best.pt \
      --name al_compare
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

# 掩码 IoU 落在 [匹配阈值, 免修改阈值) 之间时，算「需要拖一下边界」
IOU_MATCH = 0.5
IOU_FREE = 0.75

# 操作权重：等权计数会把「从零画一片病斑」和「点一下删掉一个误检」算成一样重，
# 但实际耗时差一个量级。以「新画一片」为 1 个单位工作量，其余按 M4 界面上的
# 实际操作折算。这组数要写进报告，并做敏感性分析（见 --weight-preset）。
WEIGHT_PRESETS = {
    "flat": {"新画": 1.0, "删实例": 1.0, "改类别": 1.0,
             "拖边界": 1.0, "标为无目标": 1.0},
    "realistic": {"新画": 1.0, "删实例": 0.15, "改类别": 0.15,
                  "拖边界": 0.3, "标为无目标": 0.15},
    "lenient": {"新画": 1.0, "删实例": 0.08, "改类别": 0.08,
                "拖边界": 0.2, "标为无目标": 0.08},
}


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def load_yolo_seg(label_path, w, h):
    """YOLO 分割标签 -> [(cls, mask)]。缺文件视为「这张图没有目标」。"""
    out = []
    p = Path(label_path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        xy = np.array([float(v) for v in parts[1:]], np.float32).reshape(-1, 2)
        xy[:, 0] *= w
        xy[:, 1] *= h
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(xy).astype(np.int32)], 1)
        if m.sum() == 0:
            continue
        out.append((cls, m))
    return out


def preds_to_masks(result, w, h, conf):
    """ultralytics Result -> [(cls, mask)]，过滤低置信度。"""
    out = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    boxes = result.boxes
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    keep = conf_arr >= conf
    if not keep.any():
        return out
    if result.masks is None or result.masks.data is None:
        # 只有框没有掩码（不该发生，兜底把框填成矩形）
        xyxy = boxes.xyxy.cpu().numpy()
        for i in np.where(keep)[0]:
            x1, y1, x2, y2 = xyxy[i]
            m = np.zeros((h, w), np.uint8)
            m[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = 1
            out.append((int(cls_arr[i]), m))
        return out
    md = result.masks.data.cpu().numpy()
    for i in np.where(keep)[0]:
        m = cv2.resize(md[i].astype(np.float32), (w, h),
                       interpolation=cv2.INTER_NEAREST)
        m = (m > 0.5).astype(np.uint8)
        if m.sum() == 0:
            continue
        out.append((int(cls_arr[i]), m))
    return out


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def edit_ops(gt, pred):
    """算一张图的编辑清单。返回 (编辑次数, 明细 Counter, 匹配对 IoU 列表)。"""
    ops = Counter()
    matched_ious = []
    if not gt and not pred:
        return 0, ops, matched_ious
    if gt and not pred:
        ops["新画"] = len(gt)
        return len(gt), ops, matched_ious
    if pred and not gt:
        # 真值空、预测非空：标注员先要点「无目标」，再把这些删干净
        ops["标为无目标"] = 1
        return 1, ops, matched_ious

    # 同类内贪心匹配：反复取全局最大 IoU 的一对
    pairs = []
    for i, (gc, gm) in enumerate(gt):
        for j, (pc, pm) in enumerate(pred):
            if gc != pc:
                continue
            v = iou(gm, pm)
            if v >= IOU_MATCH:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)

    used_g, used_p = set(), set()
    for v, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched_ious.append(v)
        if v < IOU_FREE:
            ops["拖边界"] += 1

    # 剩下没配上的：按 IoU 判断是「框对了类错了」还是纯粹的漏检/误检
    rest_g = [i for i in range(len(gt)) if i not in used_g]
    rest_p = [j for j in range(len(pred)) if j not in used_p]
    for i in rest_g:
        best_j, best_v = None, 0.0
        for j in rest_p:
            if j in used_p:
                continue
            v = iou(gt[i][1], pred[j][1])
            if v > best_v:
                best_v, best_j = v, j
        if best_j is not None and best_v >= IOU_MATCH:
            ops["改类别"] += 1
            used_p.add(best_j)
        else:
            ops["新画"] += 1
    for j in rest_p:
        if j not in used_p:
            ops["删实例"] += 1

    return sum(ops.values()), ops, matched_ious


def resolve_split(data_yaml, split):
    """从 data.yaml 里读出某个划分的图片目录。"""
    import yaml
    with open(data_yaml, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    root = Path(d.get("path", Path(data_yaml).parent))
    rel = d.get(split)
    if isinstance(rel, list):
        dirs = [root / r for r in rel]
    else:
        dirs = [root / rel]
    imgs = []
    for dd in dirs:
        if dd.is_dir():
            imgs += sorted(p for p in dd.iterdir()
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    return imgs


def images_to_labels(imgs, labels_dir=None):
    """默认 images/xxx/a.jpg -> labels/xxx/a.txt。

    给了 labels_dir 时改为 <labels_dir>/<stem>.txt，用于三臂对照实验的
    标注产物（out/annotations/<arm>/，见 docs/INTERFACE_ADDENDUM.md §4.3）。
    """
    if labels_dir is not None:
        d = Path(labels_dir)
        return [d / (Path(p).stem + ".txt") for p in imgs]
    out = []
    for p in imgs:
        parts = list(p.parts)
        for k in range(len(parts) - 1, -1, -1):
            if parts[k] == "images":
                parts[k] = "labels"
                break
        out.append(Path(*parts).with_suffix(".txt"))
    return out


def eval_one(weights, imgs, labs, names, conf, iou_thr, device, imgsz, max_n,
             wmap=None, pure_manual=None):
    from ultralytics import YOLO
    model = YOLO(weights)
    n = len(imgs) if not max_n else min(max_n, len(imgs))
    tot_edit = 0
    tot_gt = 0
    tot_pred = 0
    ops_all = Counter()
    miou_all = []
    per_img = []

    for k in range(n):
        ip, lp = imgs[k], labs[k]
        bgr = imread_unicode(ip)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt = load_yolo_seg(lp, w, h)
        res = model.predict(bgr, imgsz=imgsz, conf=conf, iou=iou_thr,
                            device=device, verbose=False)
        pred = preds_to_masks(res[0], w, h, conf)
        ne, ops, mi = edit_ops(gt, pred)
        tot_edit += ne
        tot_gt += len(gt)
        tot_pred += len(pred)
        miou_all += mi
        ops_all.update(ops)
        per_img.append({"image": ip.name, "n_gt": len(gt), "n_pred": len(pred),
                        "edits": ne, **{f"op_{a}": b for a, b in ops.items()}})
        if (k + 1) % 100 == 0:
            print(f"    {k+1}/{n}  累计编辑 {tot_edit}")

    w = wmap or WEIGHT_PRESETS["flat"]
    weighted = sum(c * w.get(k, 1.0) for k, c in ops_all.items())
    return {
        "weights": str(weights),
        "n_images": n,
        "n_gt_instances": tot_gt,
        "n_pred_instances": tot_pred,
        "total_edits": tot_edit,
        "edits_per_image": round(tot_edit / max(n, 1), 3),
        "edits_per_instance": round(tot_edit / max(tot_gt, 1), 3),
        # 加权工作量：以「新画一片」为 1 个单位
        "weighted_per_image": round(weighted / max(n, 1), 3),
        # 纯手工基准：每实例从零画一笔，不做任何预标注
        "manual_per_image": round(tot_gt / max(n, 1), 3),
        "weighted_vs_manual": round(
            (1 - (weighted / max(n, 1)) / (tot_gt / max(n, 1))) * 100, 1)
        if tot_gt else None,
        "weight_preset": w,
        # 匹配上的实例：掩码重合度。这就是「预标注掩码本身好不好用」的直接答案
        "matched_mask_iou": round(float(np.mean(miou_all)), 4) if miou_all else None,
        "n_matched": len(miou_all),
        "instance_recall": round(len(miou_all) / max(tot_gt, 1), 4),
        "ops": dict(ops_all),
        "per_image": per_img,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", nargs="+", required=True,
                    help="一个或多个权重，用于横向对比")
    ap.add_argument("--names", nargs="*", default=None,
                    help="与 --weights 对应的显示名；不给就用目录名")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS 的 IoU 阈值")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-n", type=int, default=0, help="只评估前 N 张，调试用")
    ap.add_argument("--name", default="editcost")
    ap.add_argument("--weight-preset", default="realistic",
                    choices=list(WEIGHT_PRESETS),
                    help="操作权重方案。flat 是等权计数，realistic/lenient 按实际操作耗时折算")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--labels-dir", default=None,
                    help="按 <stem>.txt 从该目录读标签（用于 out/annotations/<arm>/）；"
                         "给了就只评估该目录里存在标注的图")
    args = ap.parse_args()

    imgs = resolve_split(args.data, args.split)
    if not imgs:
        raise SystemExit(f"{args.data} 里 {args.split} 没有图片")
    labs = images_to_labels(imgs, args.labels_dir)
    if args.labels_dir:
        keep = [(i, l) for i, l in zip(imgs, labs) if l.exists()]
        if not keep:
            raise SystemExit(
                f"{args.labels_dir} 下没有与 {args.split} 匹配的 .txt 标注")
        imgs = [i for i, _ in keep]
        labs = [l for _, l in keep]
        print(f"{args.split}: 按 --labels-dir 过滤后 {len(imgs)} 张有标注")
    print(f"{args.split}: {len(imgs)} 张图，标签 {sum(1 for p in labs if p.exists())} 个存在")

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT / "out" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.names or [Path(w).parent.parent.name or Path(w).stem
                            for w in args.weights]
    rows = []
    wmap = WEIGHT_PRESETS[args.weight_preset]
    for w, nm in zip(args.weights, labels):
        print(f"\n[{nm}] {w}")
        r = eval_one(w, imgs, labs, None, args.conf, args.iou,
                     args.device, args.imgsz, args.max_n, wmap=wmap)
        r["label"] = nm
        rows.append(r)

    # 逐图明细落盘，方便画分布图和复核
    OP_NAMES = ["新画", "删实例", "改类别", "拖边界", "标为无目标"]
    with open(out_dir / f"{args.name}_per_image.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        keys = ["label", "image", "n_gt", "n_pred", "edits"]
        extra = [f"op_{a}" for a in OP_NAMES]
        wtr = csv.DictWriter(f, fieldnames=keys + extra, extrasaction="ignore")
        wtr.writeheader()
        for r in rows:
            for d in r["per_image"]:
                row = {"label": r["label"],
                       "image": d["image"], "n_gt": d["n_gt"],
                       "n_pred": d["n_pred"], "edits": d["edits"]}
                for a in OP_NAMES:
                    row[f"op_{a}"] = d.get(f"op_{a}", 0)
                wtr.writerow(row)

    summary = [{k: v for k, v in r.items() if k != "per_image"} for r in rows]
    with open(out_dir / f"{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 编辑成本 ===")
    print(f"{'方案':28s} {'图数':>5s} {'真值实例':>8s} {'总编辑':>7s} "
          f"{'单张':>7s} {'单实例':>7s} {'掩码IoU':>8s} {'实例召回':>8s}")
    base = rows[0]["edits_per_image"] if rows else 1
    for r in rows:
        drop = (1 - r["edits_per_image"] / base) * 100 if base else 0.0
        mi = r["matched_mask_iou"]
        print(f"{r['label'][:28]:28s} {r['n_images']:5d} {r['n_gt_instances']:8d} "
              f"{r['total_edits']:7d} {r['edits_per_image']:7.3f} "
              f"{r['edits_per_instance']:7.3f} "
              f"{(f'{mi:.3f}' if mi is not None else '-'):>8s} "
              f"{r['instance_recall']:8.3f}   相对基线 {drop:+.1f}%")

    # 加权口径：等权计数会把「新画」和「删误检」算成一样重，这里按实际耗时折算
    print(f"\n=== 加权工作量（方案 {args.weight_preset}："
          + "，".join(f"{k} {v}" for k, v in wmap.items()) + "）===")
    print(f"{'方案':28s} {'加权/张':>8s} {'纯手工/张':>10s} {'节省':>8s}")
    for r in rows:
        sv = r["weighted_vs_manual"]
        print(f"{r['label'][:28]:28s} {r['weighted_per_image']:8.3f} "
              f"{r['manual_per_image']:10.3f} "
              f"{(f'{sv:+.1f}%' if sv is not None else '-'):>8s}")
    if rows:
        print(f"\n基线方案：{rows[0]['label']}")
        print(f"操作构成：{rows[-1]['ops']}")
    print(f"\n明细：{out_dir / f'{args.name}_per_image.csv'}")
    print(f"汇总：{out_dir / f'{args.name}.json'}")


if __name__ == "__main__":
    main()

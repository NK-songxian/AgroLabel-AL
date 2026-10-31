"""M2 离线伪标注：从分类标签造出叶片实例掩码和 YOLO 分割标签。

这批数据只有「文件夹名 = 类别」，没有框也没有掩码。要训练分割模型，
就得先凭空造一批掩码出来。分两步走：

  1. 粗框：GrabCut，把四周一圈当作确定背景。这一步不追求精确，
     只要能框住叶子就行，失败时退回 LAB 距离阈值，再不行给全图框。
  2. 精细掩码：把粗框当作 box 提示送进 SAM2，它会沿着叶缘输出像素级掩码。

这一步是离线的，只跑一次，产出直接当训练数据用。

产出：
  out/classes.json              类别名 -> 类别 id（全项目共用，别各写各的）
  out/pseudo/masks/<id>.png     二值掩码 0/255
  out/pseudo/labels/<id>.txt    YOLO 分割标签：cls x1 y1 x2 y2 ...（归一化）
  out/pseudo/index.csv          每张图的面积比、框来源、点数、是否合格
  out/figures/pseudo_grid.png   16 张叠加预览

运行：
  python src/m2_pseudo.py --n 3000          # 先跑 3000 张看看
  python src/m2_pseudo.py                   # 全量，可断点续跑
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, setup_matplotlib

PROJECT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 读写
def imread_unicode(path):
    """读图。cv2.imread 在中文路径下返回 None，必须绕一下。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# ---------------------------------------------------------------- 掩码工具
def largest_component_filled(m):
    """只保留最大连通域，再把内部空洞填上。

    一张图里正常只有一片叶；填空洞是因为叶面上的白斑、病斑不该被挖掉，
    否则掩码会变成筛子，多边形也没法表示。
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros(m.shape, np.uint8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = ((lab == idx) * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return filled


def cleanup(m):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return largest_component_filled(m)


def area_ratio(m):
    return float((m > 0).sum()) / m.size


def mask_to_polygon(mask, max_points, eps_ratio):
    """掩码 -> 归一化多边形点序列。分不出来就返回 None。

    用 approxPolyDP 简化轮廓：YOLO 的标签文本不能太大，
    一条标签几万个点会把 dataloader 拖死。
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) <= 0:
        return None
    eps = eps_ratio * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    # 点太多就按周长均匀重采样，保证不超上限
    if len(approx) > max_points:
        step = len(approx) / max_points
        approx = np.array([approx[int(i * step)] for i in range(max_points)])
    if len(approx) < 3:
        return None
    h, w = mask.shape[:2]
    pts = np.stack([approx[:, 0] / w, approx[:, 1] / h], axis=1)
    return np.clip(pts, 0.0, 1.0)


# ---------------------------------------------------------------- 粗框
def rough_mask_labdist(bgr, ring):
    """在 LAB 空间里量「离背景色多远」，再 Otsu 切一刀。

    这批图背景是灰色影棚板，不是纯白，不能用白底阈值。
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bp = np.concatenate([
        lab[:ring].reshape(-1, 3), lab[-ring:].reshape(-1, 3),
        lab[:, :ring].reshape(-1, 3), lab[:, -ring:].reshape(-1, 3),
    ])
    bg = np.median(bp, axis=0)
    d = np.linalg.norm(lab - bg, axis=2)
    d8 = cv2.GaussianBlur(
        cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (5, 5), 0)
    thr, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cleanup((d8 > thr).astype(np.uint8) * 255)


def rough_mask_grabcut(bgr, b):
    """GrabCut：四周一圈确定背景，中心一块可能前景。"""
    h, w = bgr.shape[:2]
    ring = int(b["ring"])
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[int(h * b["center_lo"]):int(h * b["center_hi"]),
       int(w * b["center_lo"]):int(w * b["center_hi"])] = cv2.GC_PR_FGD
    gc[:ring, :] = cv2.GC_BGD
    gc[-ring:, :] = cv2.GC_BGD
    gc[:, :ring] = cv2.GC_BGD
    gc[:, -ring:] = cv2.GC_BGD

    sc = float(b["grabcut_scale"])
    small = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    hs, ws = small.shape[:2]
    gcs = cv2.resize(gc, (ws, hs), interpolation=cv2.INTER_NEAREST)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, gcs, None, bgm, fgm, int(b["grabcut_iters"]),
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    m = np.where((gcs == cv2.GC_FGD) | (gcs == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return cleanup(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))


def bbox_of(m, pad):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    h, w = m.shape[:2]
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return [max(0.0, x1 - px), max(0.0, y1 - py),
            min(w - 1.0, x2 + px), min(h - 1.0, y2 + py)]


def prompt_box(bgr, b, acc):
    """三级退化：GrabCut -> LAB距离 -> 全图。返回 (box, 来源, 粗掩码面积比)。"""
    h, w = bgr.shape[:2]
    pad = float(b["pad"])

    g = rough_mask_grabcut(bgr, b)
    rg = area_ratio(g)
    if acc["min_area_ratio"] * 1.5 <= rg <= 0.95:
        return bbox_of(g, pad), "grabcut", rg

    a = rough_mask_labdist(bgr, int(b["ring"]))
    ra = area_ratio(a)
    if acc["min_area_ratio"] * 2 <= ra <= 0.95:
        return bbox_of(a, pad), "labdist", ra

    # 整幅图都是叶面（贴脸特写），给全图框，让 SAM 自己判断
    return [2.0, 2.0, w - 3.0, h - 3.0], "full", ra


# ---------------------------------------------------------------- 主流程
def load_manifest_rows(path, split, limit):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["is_background"] == "1":
                continue
            if split and r["split"] != split:
                continue
            rows.append(r)
    # 按类别排序后轮流取，保证前 N 张里各类都有，不被字母序靠前的类占满
    rows.sort(key=lambda r: r["class_dir"])
    if limit and limit < len(rows):
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit)]
    return rows


def load_done(index_path):
    """断点续跑：已经合格的图不再重算。"""
    done = {}
    if not index_path.exists():
        return done
    with open(index_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            done[r["image"]] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT / "configs" / "default.yaml"))
    ap.add_argument("--n", type=int, default=0, help="最多处理多少张；0 = 全量")
    ap.add_argument("--split", default="train", help="只处理哪个 split；all = 全部")
    ap.add_argument("--device", default=None, help="覆盖配置里的 device")
    ap.add_argument("--resume", action="store_true", help="跳过 index.csv 里已合格的图")
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = cfg["pseudo"]
    acc = P["accept"]
    dev = args.device if args.device is not None else P["device"]
    if isinstance(dev, str) and dev.isdigit():
        dev = int(dev)

    manifest = PROJECT / cfg["paths"]["manifest"]
    if not manifest.exists():
        raise SystemExit("先跑 m0_inspect.py 生成 out/manifest.csv")

    split = "" if args.split == "all" else args.split
    rows = load_manifest_rows(manifest, split, args.n)
    if not rows:
        raise SystemExit("没有可处理的样本")

    out_dir = PROJECT / cfg["paths"]["pseudo"]
    mask_dir = out_dir / "masks"
    label_dir = out_dir / "labels"
    for d in (mask_dir, label_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 类别表：全项目唯一。注意要按「全量 manifest」建表，
    # 不能按本次抽样的子集，否则这次跑 3000 张、下次跑全量,id 就对不上了。
    all_classes = set()
    with open(manifest, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["is_background"] == "0":
                all_classes.add(r["class_dir"])
    classes = sorted(all_classes)
    cls_of = {c: i for i, c in enumerate(classes)}
    cls_path = PROJECT / cfg["paths"]["classes_json"]
    if cls_path.exists():
        with open(cls_path, encoding="utf-8") as f:
            classes = json.load(f)["classes"]
        cls_of = {c: i for i, c in enumerate(classes)}
    else:
        cls_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cls_path, "w", encoding="utf-8") as f:
            json.dump({"classes": classes,
                       "note": "类别 id 全项目唯一，训练和推理都用这张表"},
                      f, ensure_ascii=False, indent=2)
        print(f"写入类别表：{cls_path}（{len(classes)} 类）")

    index_path = out_dir / "index.csv"
    done = load_done(index_path) if args.resume else {}

    from ultralytics import SAM
    model = SAM(P["sam_weights"])
    print(f"SAM 权重：{P['sam_weights']}，device={dev}")

    fields = ["image", "class_dir", "cls_id", "split", "mask", "label",
              "area_ratio", "fill_ratio", "box_source", "box_ratio",
              "n_points", "ok", "reason"]
    f_index = open(index_path, "a" if args.resume and index_path.exists() else "w",
                   newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f_index, fieldnames=fields)
    if not (args.resume and index_path.exists()):
        writer.writeheader()

    tiles, n_ok, n_skip = [], 0, 0
    t_start = time.time()

    for i, r in enumerate(rows, 1):
        if r["path"] in done and done[r["path"]]["ok"] == "1":
            n_skip += 1
            continue

        bgr = imread_unicode(r["path"])
        if bgr is None:
            writer.writerow({**{k: "" for k in fields}, "image": r["path"],
                             "class_dir": r["class_dir"], "ok": 0,
                             "reason": "读图失败"})
            continue

        box, src, box_ratio = prompt_box(bgr, P["box"], acc)
        h, w = bgr.shape[:2]

        mask = np.zeros((h, w), np.uint8)
        res = model.predict(bgr, bboxes=[box], verbose=False, device=dev)
        if res and res[0].masks is not None and len(res[0].masks.data):
            md = res[0].masks.data.cpu().numpy()
            # 框模式下 SAM 会给若干候选，取面积最大的那片叶
            best = md[int(np.argmax([x.sum() for x in md]))]
            best = cv2.resize(best.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST)
            mask = (best > 0.5).astype(np.uint8) * 255
            mask = largest_component_filled(mask)

        # 掩码不得跑出提示框太多，否则是 SAM 串到别的物体上了
        pad_x, pad_y = (box[2] - box[0]) * 0.15 + 4, (box[3] - box[1]) * 0.15 + 4
        lim = np.zeros((h, w), np.uint8)
        cv2.rectangle(lim, (int(max(0, box[0] - pad_x)), int(max(0, box[1] - pad_y))),
                      (int(min(w - 1, box[2] + pad_x)), int(min(h - 1, box[3] + pad_y))),
                      255, -1)
        mask = cv2.bitwise_and(mask, lim)

        ar = area_ratio(mask)
        # 掩码占提示框的比例。正常叶子塞进框里能占到 0.5 以上；
        # 只切到叶面上一小条病斑时这个值只有 0.0x，用它抓这类错。
        box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
        fill_ratio = float((mask > 0).sum()) / box_area
        poly = None
        reason = ""
        if ar < acc["min_area_ratio"]:
            reason = "掩码过小"
        elif ar > acc["max_area_ratio"]:
            reason = "掩码过大"
        elif fill_ratio < acc["min_box_fill"]:
            reason = "掩码没填满提示框"
        else:
            poly = mask_to_polygon(mask, P["polygon"]["max_points"],
                                   P["polygon"]["eps_ratio"])
            if poly is None or len(poly) < acc["min_points"]:
                reason = "多边形点数不足"
                poly = None
        ok = poly is not None and reason == ""

        key = hashlib.md5(r["path"].encode("utf-8")).hexdigest()[:12]
        mask_file = mask_dir / f"{key}.png"
        label_file = label_dir / f"{key}.txt"
        if ok:
            imwrite_unicode(mask_file, mask)
            with open(label_file, "w", encoding="utf-8") as f:
                f.write(f"{cls_of.get(r['class_dir'], 0)} "
                        + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly))
            n_ok += 1

        writer.writerow({
            "image": r["path"], "class_dir": r["class_dir"],
            "cls_id": cls_of.get(r["class_dir"], -1), "split": r["split"],
            "mask": str(mask_file) if ok else "", "label": str(label_file) if ok else "",
            "area_ratio": round(ar, 4), "fill_ratio": round(fill_ratio, 4),
            "box_source": src,
            "box_ratio": round(box_ratio, 4),
            "n_points": len(poly) if poly is not None else 0,
            "ok": int(ok), "reason": reason,
        })
        f_index.flush()

        if len(tiles) < 16:
            ov = bgr.copy()
            ov[mask > 0] = (0.6 * ov[mask > 0] + 0.4 * np.array([60, 180, 255])).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnts, -1, (0, 0, 255), 1)
            tiles.append(ov)

        if i % 100 == 0:
            el = time.time() - t_start
            done_n = max(i - n_skip, 1)
            print(f"  {i}/{len(rows)}  合格 {n_ok}  "
                  f"{el/done_n*1000:.0f}ms/张  预计剩 {el/done_n*(len(rows)-i)/60:.0f} 分钟")

    f_index.close()

    if tiles:
        plt = setup_matplotlib()
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, t in zip(axes.ravel(), tiles):
            ax.imshow(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
        plt.suptitle("伪标注预览（蓝=掩码，红=轮廓）", fontsize=12)
        plt.tight_layout()
        plt.savefig(PROJECT / cfg["paths"]["figures"] / "pseudo_grid.png", dpi=150)
        plt.close(fig)

    total = len(rows) - n_skip
    print("\n完成。")
    print(f"  本次处理 {total} 张，合格 {n_ok} 张"
          f"（{n_ok/max(total,1)*100:.1f}%），跳过已合格 {n_skip} 张")
    print(f"  耗时 {(time.time()-t_start)/60:.1f} 分钟")
    print(f"  索引：{index_path}")


if __name__ == "__main__":
    main()

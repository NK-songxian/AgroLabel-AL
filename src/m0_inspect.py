"""M0 数据体检：把 H:\Plant 变成一张可查的表。

产出三样东西：
  out/manifest.csv   每行一张图：路径、类别、尺寸、划分
  out/figures/class_dist.png   类别分布图
  out/data_report.md 体检报告

运行：
  python src/m0_inspect.py
  python src/m0_inspect.py --no-size     # 跳过读尺寸，更快
"""
import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import iter_images, parse_class_name, load_config, setup_matplotlib

PROJECT = Path(__file__).resolve().parents[1]


def collect(data_root, sep, background_dir):
    """遍历全部图片，解析类别。返回记录列表。"""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在：{root}")

    records = []
    n = 0
    for p in iter_images(root):
        dirname = p.parent.name
        crop, disease = parse_class_name(dirname, sep)
        records.append({
            "path": str(p),
            "rel_path": str(p.relative_to(root)),
            "class_dir": dirname,
            "crop": crop,
            "disease": disease,
            "is_background": int(dirname == background_dir),
        })
        n += 1
        if n % 10000 == 0:
            print(f"  已扫 {n} 张 ...")
    return records


def read_sizes(records):
    """读每张图的宽高。PIL 只读文件头，不解码像素，很快。"""
    from PIL import Image
    bad = 0
    for i, r in enumerate(records):
        try:
            with Image.open(r["path"]) as im:
                r["width"], r["height"] = im.size
        except Exception:
            r["width"] = r["height"] = 0
            bad += 1
        if (i + 1) % 10000 == 0:
            print(f"  已读尺寸 {i+1}/{len(records)} ...")
    return bad


def stratified_split(records, cfg):
    """按类别分层切分。返回 {split: count}。

    为什么不用 sklearn：这一步只有几十行，自己写能看清每一步，
    而且以后要在"按批次划分"上加逻辑时不用跟库的接口较劲。
    """
    seed = cfg["split"]["seed"]
    frac = (cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"])
    min_for_split = cfg["split"]["min_for_split"]

    by_class = defaultdict(list)
    for r in records:
        by_class[r["class_dir"]].append(r)

    rng = random.Random(seed)
    stat = Counter()
    for cls, items in by_class.items():
        idx = list(range(len(items)))
        rng.shuffle(idx)
        if len(items) < min_for_split:
            # 小类全进训练集，避免验证/测试集出现空类
            for i in idx:
                items[i]["split"] = "train"
                stat["train"] += 1
            continue
        n_train = int(len(items) * frac[0])
        n_val = int(len(items) * frac[1])
        for k, i in enumerate(idx):
            if k < n_train:
                s = "train"
            elif k < n_train + n_val:
                s = "val"
            else:
                s = "test"
            items[i]["split"] = s
            stat[s] += 1
    return stat


def plot_dist(records, out_png, top_n=39):
    plt = setup_matplotlib()
    cnt = Counter(r["class_dir"] for r in records)
    items = cnt.most_common()
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]

    fig, ax = plt.subplots(figsize=(9, 11))
    colors = ["#c9d8ea" if "Background" in l else "#1a5fb4" for l in labels]
    ax.barh(range(len(vals)), vals, color=colors, height=0.72)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("图像数量")
    ax.set_title(f"PlantVillage 类别分布（共 {len(vals)} 类，{sum(vals):,} 张）", fontsize=12)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.005, i, str(v), va="center", fontsize=7)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close(fig)


def write_report(records, stat, bad, out_md, cfg):
    cnt = Counter(r["class_dir"] for r in records)
    vals = sorted(cnt.values())
    total = len(records)
    bg = sum(1 for r in records if r["is_background"])
    sizes = Counter(f'{r["width"]}x{r["height"]}' for r in records if r.get("width"))
    exts = Counter(Path(r["path"]).suffix.lower() for r in records)

    tomato = [k for k in cnt if k.startswith("Tomato")]
    lines = [
        "# 数据体检报告（M0）\n",
        f"- 数据根目录：`{cfg['data']['root']}`",
        f"- 图像总数：**{total:,}**",
        f"- 类别数：**{len(cnt)}**（其中背景类 {bg:,} 张）",
        f"- 最大类：`{max(cnt, key=cnt.get)}` {max(cnt.values()):,} 张",
        f"- 最小类：`{min(cnt, key=cnt.get)}` {min(cnt.values()):,} 张",
        f"- 不平衡比：**{max(cnt.values())/min(cnt.values()):.1f} : 1**",
        f"- 番茄类：{len(tomato)} 类，合计 {sum(cnt[k] for k in tomato):,} 张"
        f"（占 {sum(cnt[k] for k in tomato)/total*100:.1f}%）",
        f"- 读尺寸失败：{bad} 张\n",
        "## 划分结果\n",
        "| 划分 | 张数 | 占比 |",
        "|---|---|---|",
    ]
    for s in ("train", "val", "test"):
        lines.append(f"| {s} | {stat[s]:,} | {stat[s]/total*100:.1f}% |")

    lines += ["\n## 图像尺寸分布（前 8）\n", "| 尺寸 | 张数 |", "|---|---|"]
    for k, v in sizes.most_common(8):
        lines.append(f"| {k} | {v:,} |")

    lines += ["\n## 扩展名分布\n", "| 扩展名 | 张数 |", "|---|---|"]
    for k, v in exts.most_common():
        lines.append(f"| {k} | {v:,} |")

    lines += ["\n## 类别清单\n", "| 类别目录 | 张数 | 作物 | 病害 |", "|---|---|---|---|"]
    crop_of = {}
    dis_of = {}
    for r in records:
        crop_of[r["class_dir"]] = r["crop"]
        dis_of[r["class_dir"]] = r["disease"]
    for k, v in cnt.most_common():
        lines.append(f"| `{k}` | {v:,} | {crop_of[k]} | {dis_of[k]} |")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT / "configs" / "default.yaml"))
    ap.add_argument("--no-size", action="store_true", help="跳过读图像尺寸")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = PROJECT / cfg["paths"]["out"]
    fig_dir = PROJECT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 遍历目录 ...")
    records = collect(cfg["data"]["root"], cfg["classes"]["separator"],
                      cfg["classes"]["background_dir"])
    print(f"      共 {len(records):,} 张")

    if args.no_size:
        for r in records:
            r["width"] = r["height"] = 0
        bad = 0
    else:
        print("[2/5] 读取尺寸 ...")
        bad = read_sizes(records)

    print("[3/5] 分层划分 ...")
    stat = stratified_split(records, cfg)
    print(f"      {dict(stat)}")

    print("[4/5] 写 manifest.csv ...")
    manifest = PROJECT / cfg["paths"]["manifest"]
    cols = ["path", "rel_path", "class_dir", "crop", "disease",
            "is_background", "width", "height", "split"]
    with open(manifest, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in cols})
    print(f"      {manifest}")

    print("[5/5] 出图与报告 ...")
    plot_dist(records, fig_dir / "class_dist.png")
    write_report(records, stat, bad, PROJECT / "out" / "data_report.md", cfg)

    print("\n完成。")
    print(f"  清单：{manifest}")
    print(f"  分布图：{fig_dir / 'class_dist.png'}")
    print(f"  报告：{PROJECT / 'out' / 'data_report.md'}")


if __name__ == "__main__":
    main()

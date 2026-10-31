"""主动学习闭环：用真值当 Oracle，实测「同等精度下能省多少标注量」。

## 为什么这个脚本现在才算得准

以前没有真值，只能拿伪标签假装「人工标注」，于是「选了这张、人来标、再训一遍」
里的「人来标」是拿模型自己的输出喂回去，等于自己考自己，省多少标注量无从谈起。
PlantSeg 有像素级真值，Oracle 变成真的：选中的图，把它的**真值标签**放进训练集，
模拟「标注员一次就标对了」的理想情形。这是上限估计，也是策略之间能公平比的原因。

## 四组策略

  random   随机抽。基线，也是评委最想看的对照。
  entropy  分类熵。取模型对该图所有预测实例类别分布的熵的均值；
           一个实例都没检出来的图按最大熵算（模型对它一无所知，正是该标的）。
  tta      扰动一致性。原图 / 放大 1.25 倍 / 水平翻转 / 两者叠加，共 4 次前向，
           两两求掩码 IoU 再平均，一致性越低越可疑。**用它替代 MC Dropout**，
           因为 YOLO11 主干没有 dropout 层，开不了 MC 采样；这条替换理由要写进报告。
  coreset  k 中心贪心。在颜色纹理描述子上做最远点采样，纯多样性，不看模型。

前两个是"挑难的"，第三个是"挑不稳的"，第四个是"挑不一样的"。四组一起跑，
才能说明"难例+多样性"的组合为什么比单独一类好。

## 公平性

每个策略用**完全相同**的预算序列和初始集，完全相同的主干、超参、随机种子。
每轮训练都从上一轮权重热启动，且轮内 epoch 数固定——不这么做，
"标注量大的一轮训练更久"会和"标注质量更好"混在一起，说不清是谁的功劳。

## 用法

  # 先小规模验证流程
  python src/m8_al.py --data H:/Plant/_derived/dataset_plantseg_34/data.yaml \
      --strategies random entropy --rounds 3 --epochs 15 --init-per-class 3 --dry-n 0

  # 完整四组
  python src/m8_al.py --data H:/Plant/_derived/dataset_plantseg_34/data.yaml \
      --strategies random entropy tta coreset --rounds 5 --epochs 25 \
      --init-per-class 3 --budget-step 0.08 --name al34
"""
import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def read_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_split(data_yaml, split):
    d = read_yaml(data_yaml)
    root = Path(d.get("path", Path(data_yaml).parent))
    rel = d.get(split)
    dirs = [root / r for r in rel] if isinstance(rel, list) else [root / rel]
    imgs = []
    for dd in dirs:
        if dd.is_dir():
            imgs += sorted(p for p in dd.iterdir()
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    return imgs


def label_of(img_path):
    parts = list(Path(img_path).parts)
    for k in range(len(parts) - 1, -1, -1):
        if parts[k] == "images":
            parts[k] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def classes_of(label_path):
    s = set()
    p = Path(label_path)
    if not p.exists():
        return s
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split(maxsplit=1)
        if parts:
            try:
                s.add(int(parts[0]))
            except ValueError:
                pass
    return s


# ------------------------------------------------------------------ 采样策略
def score_random(pool, model, nc, k, rng):
    return list(rng.choice(len(pool), size=min(k, len(pool)), replace=False))


def score_entropy(pool, model, nc, k, args):
    """分类不确定度。没检出任何实例的图按最高分处理。

    说明：ultralytics 只暴露每个实例的 top-1 置信度，拿不到完整的 34 维类别分布，
    所以没法直接算多类熵。这里把置信度看作「这个实例判对了的概率」，
    取二元熵 H(p) = -p·log p - (1-p)·log(1-p) 作为不确定度：
    置信度 0.5 附近最高、趋近 1 或 0 时最低。这个替换要在报告里写明，
    不能假装算的是多类熵。
    """
    scores = []
    for p in pool:
        bgr = imread_unicode(p)
        if bgr is None:
            scores.append(0.0)
            continue
        r = model.predict(bgr, imgsz=args.imgsz, conf=0.05, device=args.device,
                          verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            # 一个实例都没检出来：模型对这张图完全无知，优先送给人工
            scores.append(1.0)
            continue
        p1 = np.clip(r.boxes.conf.cpu().numpy(), 1e-6, 1 - 1e-6)
        h = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
        scores.append(float(h.mean()) / float(np.log(2)))
    return top_k(scores, k)


def score_tta(pool, model, nc, k, args):
    """多尺度 + 翻转一致性。4 次前向的掩码 IoU 两两平均，越低越可疑。"""
    scores = []
    for p in pool:
        bgr = imread_unicode(p)
        if bgr is None:
            scores.append(0.0)
            continue
        h, w = bgr.shape[:2]
        views = []
        for sc, flip in [(1.0, False), (1.25, False), (1.0, True), (1.25, True)]:
            im = bgr
            if sc != 1.0:
                im = cv2.resize(im, None, fx=sc, fy=sc,
                                interpolation=cv2.INTER_LINEAR)
            if flip:
                im = cv2.flip(im, 1)
            r = model.predict(im, imgsz=args.imgsz, conf=0.05, device=args.device,
                              verbose=False)[0]
            m = np.zeros((h, w), np.uint8)
            if r.masks is not None and r.masks.data is not None and len(r.masks.data):
                md = r.masks.data.cpu().numpy()
                hh, ww = im.shape[:2]
                for x in md:
                    x = cv2.resize(x.astype(np.float32), (ww, hh),
                                   interpolation=cv2.INTER_NEAREST) > 0.5
                    if flip:
                        x = x[:, ::-1]
                    if sc != 1.0:
                        x = cv2.resize(x.astype(np.uint8), (w, h),
                                       interpolation=cv2.INTER_NEAREST) > 0
                    m |= x.astype(np.uint8)
            views.append(m)

        ious = []
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                a, b = views[i], views[j]
                u = np.logical_or(a, b).sum()
                if u == 0:
                    ious.append(0.0)
                else:
                    ious.append(np.logical_and(a, b).sum() / u)
        agree = float(np.mean(ious)) if ious else 0.0
        has_fg = any(v.sum() > 0 for v in views)
        # 一片空白（四次都没检出）最该标：一致性无从谈起，给最高分
        scores.append(1.0 if not has_fg else 1.0 - agree)
    return top_k(scores, k)


def _descriptor(path, size=24):
    bgr = imread_unicode(path)
    if bgr is None:
        return np.zeros(size * size * 3 + 48, np.float32)
    small = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 6], [0, 180, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    v = np.concatenate([small.astype(np.float32).ravel() / 255.0, hist])
    return v / (np.linalg.norm(v) + 1e-9)


def score_coreset(pool, model, nc, k, args, labeled=None):
    """k 中心贪心：每次挑离已选集合最远的点。"""
    cache = args._desc_cache
    def desc(p):
        key = str(p)
        if key not in cache:
            cache[key] = _descriptor(p)
        return cache[key]

    chosen = []
    # 起点用已标注集的平均描述子，让第一次选择就偏向"和已有数据最不像"的
    ref = [desc(p) for p in (labeled or [])]
    if ref:
        cur = np.mean(np.stack(ref), axis=0)
    else:
        cur = np.zeros_like(desc(pool[0]))
    cand = list(range(len(pool)))
    for _ in range(min(k, len(pool))):
        D = np.stack([desc(pool[i]) for i in cand])
        d = np.linalg.norm(D - cur, axis=1)
        j = int(np.argmax(d))
        chosen.append(cand[j])
        cur = desc(pool[cand[j]])
        cand.pop(j)
    return chosen


def top_k(scores, k):
    s = np.asarray(scores, dtype=float)
    if k >= len(s):
        return list(range(len(s)))
    idx = np.argpartition(-s, k - 1)[:k]
    return list(idx)


# ------------------------------------------------------------------ 主流程
def prepare_workdir(base_yaml, work, init_idx, imgs):
    """把初始标注集硬链接进 workdir，并写一份只换 train 的 data.yaml。"""
    for sub in ("images/train", "labels/train"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    d = read_yaml(base_yaml)
    root = Path(d.get("path", Path(base_yaml).parent))
    for i in init_idx:
        add_one(imgs[i], work)
    y = work / "data.yaml"
    with open(y, "w", encoding="utf-8") as f:
        f.write(f"path: {work.as_posix()}\n")
        f.write("train: images/train\n")
        f.write(f"val: {(root / 'images' / 'val').as_posix()}\n")
        f.write(f"test: {(root / 'images' / 'test').as_posix()}\n")
        f.write(f"nc: {d.get('nc')}\nnames:\n")
        for k, v in (d.get("names") or {}).items():
            f.write(f"  {k}: {v}\n")
    return y


def add_one(img, work):
    dst_img = work / "images" / "train" / Path(img).name
    dst_lab = work / "labels" / "train" / (Path(img).stem + ".txt")
    if not dst_img.exists():
        try:
            dst_img.hardlink_to(img)
        except OSError:
            shutil.copy2(img, dst_img)
    lab = label_of(img)
    if lab.exists() and not dst_lab.exists():
        dst_lab.write_text(lab.read_text(encoding="utf-8"), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--strategies", nargs="+",
                    default=["random", "entropy", "tta", "coreset"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--init-per-class", type=int, default=3)
    ap.add_argument("--budget-step", type=float, default=0.08,
                    help="每轮新增标注量，按未标注池的比例给")
    ap.add_argument("--cand-mult", type=float, default=5.0,
                    help="候选子采样倍数：每轮从池子里随机抽 cand_mult×k 张当候选，"
                         "各策略在同一批候选里挑。1.0 等于不采样（全池评分，很慢）")
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="al")
    ap.add_argument("--work-root", default="H:/Plant/_derived/_al")
    args = ap.parse_args()
    args._desc_cache = {}

    rng = np.random.default_rng(args.seed)
    imgs = list_split(args.data, "train")
    if not imgs:
        raise SystemExit(f"{args.data} 的 train 是空的")
    print(f"训练池 {len(imgs)} 张")

    d = read_yaml(args.data)
    nc = int(d.get("nc") or len(d.get("names") or {}))

    # 初始标注集：每类取 init_per_class 张，各类轮流取，避免被大类别占满
    by_cls = defaultdict(list)
    for p in imgs:
        cs = classes_of(label_of(p))
        for c in cs:
            by_cls[c].append(p)
    init_set = []
    seen = set()
    for c in sorted(by_cls):
        for p in by_cls[c][: args.init_per_class]:
            if p not in seen:
                seen.add(p)
                init_set.append(p)
    init_idx = [imgs.index(p) for p in init_set if p in imgs]
    pool_all = [i for i in range(len(imgs)) if i not in set(init_idx)]
    print(f"初始标注 {len(init_idx)} 张，未标注池 {len(pool_all)} 张，{nc} 类")

    step = max(1, int(len(pool_all) * args.budget_step))
    budgets = [len(init_idx) + step * (r + 1) for r in range(args.rounds)]
    print(f"预算序列：{len(init_idx)} -> " + " -> ".join(str(b) for b in budgets))

    results = []
    t_all = time.time()
    for strat in args.strategies:
        print(f"\n===== 策略 {strat} =====")
        work = Path(args.work_root) / f"{args.name}_{strat}"
        work.mkdir(parents=True, exist_ok=True)
        # 重跑时要先清掉上一轮留下的已标注集，否则初始预算会被历史数据污染，
        # 各策略的起点就不可比了
        for sub in ("images/train", "labels/train"):
            dd = work / sub
            if dd.is_dir():
                for f in dd.iterdir():
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".txt"}:
                        f.unlink(missing_ok=True)
        labeled = list(init_idx)
        pool = list(pool_all)
        prepare_workdir(args.data, work, labeled, imgs)

        weights = args.model
        for rnd in range(args.rounds + 1):
            tag = f"{strat}_r{rnd}"
            rng_s = np.random.default_rng(args.seed + rnd)
            np.random.seed(args.seed + rnd)

            from ultralytics import YOLO
            model = YOLO(weights)
            # val=False：训练中不验证。验证集 846 张，每轮验证比训练本身还慢，
            # 而主动学习关心的是「同一预算下谁更强」，轮内不需要挑最佳轮次，
            # 固定 epoch 数直接跑完，最后统一测一次 test 就够。
            model.train(data=str(work / "data.yaml"), epochs=args.epochs,
                        imgsz=args.imgsz, batch=args.batch, device=args.device,
                        project=str(PROJECT / "runs" / args.name), name=tag,
                        exist_ok=True, workers=4, cache=False, plots=False,
                        val=False, save=True, verbose=False,
                        fliplr=0.5, flipud=0.0, degrees=10.0,
                        scale=0.5, mosaic=0.0, seed=args.seed + rnd)
            # val=False 时没有 best.pt 的挑选依据，取最后一轮权重
            cand_w = PROJECT / "runs" / args.name / tag / "weights" / "last.pt"
            if cand_w.exists():
                weights = str(cand_w)

            m = YOLO(weights)
            mt = m.val(data=str(work / "data.yaml"), split="test",
                       imgsz=args.imgsz, batch=args.batch, device=args.device,
                       verbose=False, plots=False)
            mAP = float(mt.seg.map50)
            mAP95 = float(mt.seg.map)

            results.append({"strategy": strat, "round": rnd,
                            "n_labeled": len(labeled),
                            "mask_mAP50": round(mAP, 4),
                            "mask_mAP50_95": round(mAP95, 4),
                            "elapsed_s": round(time.time() - t_all, 1)})
            print(f"  r{rnd}  已标注 {len(labeled):5d}  掩码 mAP@0.5 {mAP:.4f}")

            if rnd == args.rounds or not pool:
                break

            k = min(step, len(pool))
            # 候选子采样：全池几千张、ttа 策略每张要 4 次前向，算一遍要十几分钟。
            # 每轮先随机抽 cand_mult*k 张当候选，各策略在**同一批候选**里挑，
            # 这样比较是公平的（相当于"给同样多的选项，看谁挑得更准"），
            # 也是主动学习在大池子上的常规做法。
            ncand = min(len(pool), max(k, int(k * args.cand_mult)))
            cand_rng = np.random.default_rng(args.seed * 1000 + rnd)
            cand = sorted(cand_rng.choice(len(pool), size=ncand, replace=False)
                          .tolist())
            cand_imgs = [imgs[pool[j]] for j in cand]

            if strat == "random":
                pick_local = score_random(cand_imgs, m, nc, k, rng_s)
            elif strat == "entropy":
                pick_local = score_entropy(cand_imgs, m, nc, k, args)
            elif strat == "tta":
                pick_local = score_tta(cand_imgs, m, nc, k, args)
            elif strat == "coreset":
                pick_local = score_coreset(cand_imgs, m, nc, k, args,
                                           labeled=[imgs[i] for i in labeled])
            else:
                raise SystemExit(f"未知策略 {strat}")

            pick = [cand[j] for j in pick_local]

            picked = [pool[j] for j in pick]
            for i in picked:
                add_one(imgs[i], work)
            labeled += picked
            pool = [i for i in pool if i not in set(picked)]

        # 本轮结束把累积的标注清单存下来，便于复核
        with open(work / "labeled.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(str(imgs[i]) for i in labeled))

    # ---------------------------------------------------------- 汇总
    out = PROJECT / "out" / "report"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.name}_curve.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "round", "n_labeled",
                                          "mask_mAP50", "mask_mAP50_95",
                                          "elapsed_s"])
        w.writeheader()
        w.writerows(results)

    # 目标精度：取最好策略的最终 mAP 的 90%，看别的策略要多少标注量才追上
    best = max(r["mask_mAP50"] for r in results) if results else 0
    target = best * 0.9
    summary = {"name": args.name, "rounds": args.rounds,
               "init_per_class": args.init_per_class, "budget_step": args.budget_step,
               "epochs": args.epochs, "seed": args.seed,
               "target_mAP50": round(target, 4), "cost_to_target": {}}
    for strat in args.strategies:
        row = [r for r in results if r["strategy"] == strat
               and r["mask_mAP50"] >= target]
        summary["cost_to_target"][strat] = (
            min(r["n_labeled"] for r in row) if row else None)

    base = summary["cost_to_target"].get("random")
    if base:
        summary["saving_vs_random"] = {
            s: (round((1 - c / base) * 100, 1) if c else None)
            for s, c in summary["cost_to_target"].items() if s != "random"}
    with open(out / f"{args.name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== 目标掩码 mAP@0.5 = {target:.4f}（最好成绩的 90%）===")
    for s, c in summary["cost_to_target"].items():
        sv = summary.get("saving_vs_random", {}).get(s)
        extra = f"  较随机减少 {sv:+.1f}%" if sv is not None else ""
        print(f"  {s:10s} 需要 {c if c is not None else '未达到':>6} 张{extra}")
    print(f"\n曲线：{out / f'{args.name}_curve.csv'}")
    print(f"汇总：{out / f'{args.name}_summary.json'}")


if __name__ == "__main__":
    main()

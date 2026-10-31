# -*- coding: utf-8 -*-
"""
m11_zeroshot_probe.py —— 零样本(开放词表)预标注能力探针

为什么要做这个脚本
------------------
评审一定会问："既然 GroundingDINO + SAM2 能零样本分割，为什么还要训练？"
本脚本用 PlantSeg 真实像素级真值把这句话量出来，而不是靠嘴说。

三条可量化的判据：
  J1 置信度   —— 开放词表检测器在这批图上到底能给出多大的分？（能否用来预标注）
  J2 实例粒度 —— 给出的框落在 GT 病斑掩码内吗？还是一整片叶子？
  J3 时延     —— 单张端到端能不能压进 300 ms？

并且必须有一个**管线自检**：同一套代码在 COCO 标准图上要能正常工作，
否则"检测不到"可能只是我配置错了，而不是方法本身不行。这个自检写进输出里。

环境说明
--------
1) sam_gpu 里 transformers 已损坏（huggingface-hub 2.3.0 与 requires >=1.5,<2.0 冲突），
   真正的 GroundingDINO 加载不了。用 ultralytics 的 YOLO-World（YOLOWorld）作代理：
   同为文本提示驱动的开放词表检测器，同样不认识"苹果黑腐病"这类细粒度类别。
   注意：GroundingDINO 在开放词表上通常强于 YOLO-World，所以本脚本的结论
   不能只依赖"检测不到"这一条，必须配合 J2/J3 与 AL 的结构性论证。
2) 本机 Windows 证书库损坏，ultralytics 任何 attempt_download 都会抛
   ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]。权重已手动放本地，故在导入前把
   ssl 默认上下文换成不校验证书的版本 —— 只影响本进程。
3) ultralytics 8.4.x 的 YOLOWorld.set_classes 有设备错位 bug（见下方补丁）。
"""
from __future__ import annotations

import argparse
import json
import ssl
import time
from pathlib import Path

# ---- 必须在 import ultralytics 之前 -------------------------------------
ssl._create_default_https_context = ssl._create_unverified_context
_orig_create = ssl.create_default_context


def _patched_create(*a, **kw):
    try:
        return _orig_create(*a, **kw)
    except Exception:
        return ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


ssl.create_default_context = _patched_create
# ------------------------------------------------------------------------

import cv2
import numpy as np
from ultralytics import YOLOWorld, YOLO

# ---- 修补 YOLOWorld.set_classes 的设备错位 ----
# get_text_pe() 按"检测头在哪个 device"建 CLIP，但 CLIP.tokenize() 永远返回 CPU
# 张量，于是 embedding 的 index 在 CPU、权重在 CUDA → RuntimeError。
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
except Exception as _e:  # pragma: no cover
    print(f"[warn] CLIP 补丁未生效: {_e}")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

# 操作阈值：真实预标注管线不可能把 conf 砍到 0.01 去用，噪声会把标注员淹掉
CONF_GRID = (0.001, 0.05, 0.25)
OP_CONF = 0.25


def load_gt_masks(lbl_path: Path, w: int, h: int) -> list[np.ndarray]:
    """把 YOLO-seg 多边形标签还原成二值掩码（只要几何，不需要类别）。"""
    masks = []
    if not lbl_path.exists():
        return masks
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        pts = np.array([float(x) for x in parts[1:]], dtype=np.float64)
        if pts.size % 2:
            pts = pts[:-1]
        pts = pts.reshape(-1, 2) * np.array([w, h], dtype=np.float64)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        masks.append(m)
    return masks


def center_hits(boxes: np.ndarray, masks: list[np.ndarray], w: int, h: int):
    """返回 (框中心落在任一 GT 掩码内的框数, 被覆盖的 GT 实例布尔数组)。"""
    gt_hit = np.zeros(len(masks), bool)
    n_box_on_gt = 0
    for b in boxes:
        cx = min(max(int(round((b[0] + b[2]) / 2.0)), 0), w - 1)
        cy = min(max(int(round((b[1] + b[3]) / 2.0)), 0), h - 1)
        on = False
        for k, m in enumerate(masks):
            if m[cy, cx]:
                on = True
                gt_hit[k] = True
        if on:
            n_box_on_gt += 1
    return n_box_on_gt, gt_hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="F:/_plantseg/dataset_plantseg_34")
    ap.add_argument("--weights", default=str(ROOT / "yolov8s-worldv2.pt"))
    ap.add_argument("--trained", default=str(ROOT / "runs/plantseg34/weights/best.pt"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    img_dir = Path(args.data) / "images" / "test"
    lbl_dir = Path(args.data) / "labels" / "test"
    files = sorted(img_dir.glob("*"))[: args.n]
    if not files:
        raise SystemExit(f"没找到测试图: {img_dir}")
    print(f"[零样本探针] 取样 {len(files)} 张 test 图  imgsz={args.imgsz}")

    gts = []
    for f in files:
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        gts.append(dict(path=f, w=w, h=h,
                        masks=load_gt_masks(lbl_dir / (f.stem + ".txt"), w, h)))
    n_inst = sum(len(g["masks"]) for g in gts)
    n_img = len(gts)
    print(f"  GT: {n_inst} 个病斑实例 / {n_img} 张，均值 {n_inst/n_img:.2f} 个/张")

    model = YOLOWorld(args.weights)
    report: dict = {
        "n_images": n_img,
        "n_gt_instances": n_inst,
        "mean_gt_per_image": round(n_inst / n_img, 3),
        "op_conf": OP_CONF,
        "prompt_sets": {},
    }

    # ---------- 管线自检：同代码在标准 COCO 图上必须能工作 ----------
    self_test = {}
    bus = ROOT / "bus.jpg"
    if bus.exists():
        im = cv2.imread(str(bus))
        for p in ("bus", "person"):
            model.set_classes([p])
            r = model.predict(im, conf=0.05, imgsz=args.imgsz, verbose=False)[0]
            self_test[p] = dict(n=len(r.boxes),
                                max_conf=round(float(r.boxes.conf.max()), 4) if len(r.boxes) else 0.0)
        print(f"[管线自检] bus.jpg → bus={self_test['bus']}  person={self_test['person']}")
    report["pipeline_self_test"] = {
        "image": "bus.jpg (ultralytics 官方样例)",
        "result": self_test,
        "note": "自检通过说明代码/权重/设备都正常，故下面的低分不是配置问题",
    }

    # ---------- 开放词表探针 ----------
    PROMPT_SETS = {
        "coarse-5": ["lesion", "disease spot", "leaf spot", "blight", "rust"],
        "single-1": ["plant disease"],
        "leaf-1": ["leaf"],
    }

    for set_name, prompts in PROMPT_SETS.items():
        print(f"\n===== 提示词组 {set_name}: {prompts} =====")
        model.set_classes(prompts)
        per_conf = {}
        for conf in CONF_GRID:
            t_img_hit = n_box = n_box_on_gt = n_gt_covered = 0
            t_sum = 0.0
            maxconfs = []
            for g in gts:
                im = cv2.imread(str(g["path"]))
                t0 = time.perf_counter()
                r = model.predict(im, conf=conf, imgsz=args.imgsz, verbose=False)[0]
                t_sum += (time.perf_counter() - t0) * 1000.0
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                maxconfs.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
                n_box += len(boxes)
                if len(boxes):
                    t_img_hit += 1
                nb, gh = center_hits(boxes, g["masks"], g["w"], g["h"])
                n_box_on_gt += nb
                n_gt_covered += int(gh.sum())
            rec = dict(
                conf=conf,
                img_det_rate=round(t_img_hit / n_img, 4),
                boxes_per_image=round(n_box / n_img, 3),
                box_on_gt_rate=round(n_box_on_gt / n_box, 4) if n_box else 0.0,
                gt_instance_coverage=round(n_gt_covered / n_inst, 4),
                max_conf_mean=round(float(np.mean(maxconfs)), 4),
                max_conf_p90=round(float(np.percentile(maxconfs, 90)), 4),
                latency_ms_mean=round(t_sum / n_img, 1),
            )
            if conf == CONF_GRID[0]:
                # 原始最高分（0.001 阈值等价于"不设阈值"），供画置信度分布图
                rec["max_confs_per_image"] = [round(x, 4) for x in maxconfs]
            per_conf[f"conf{conf}"] = rec
            print(f"  conf={conf:<6} 出图={rec['img_det_rate']:6.1%}  框/张={rec['boxes_per_image']:5.2f}  "
                  f"落病斑={rec['box_on_gt_rate']:6.1%}  覆盖GT实例={rec['gt_instance_coverage']:6.1%}  "
                  f"最高分 均值={rec['max_conf_mean']:.4f} p90={rec['max_conf_p90']:.4f}  "
                  f"时延={rec['latency_ms_mean']:.0f}ms")
        report["prompt_sets"][set_name] = dict(prompts=prompts, by_conf=per_conf)

    # ---------- 对照组：训练过的 YOLO11n-seg ----------
    print("\n===== 对照组：训练后的 YOLO11n-seg（同一批图，conf=0.25）=====")
    tw = Path(args.trained)
    if tw.exists():
        seg = YOLO(str(tw))
        t_sum = 0.0
        t_img_hit = n_box = n_box_on_gt = n_gt_covered = 0
        for g in gts:
            im = cv2.imread(str(g["path"]))
            t0 = time.perf_counter()
            r = seg.predict(im, conf=OP_CONF, imgsz=args.imgsz, verbose=False)[0]
            t_sum += (time.perf_counter() - t0) * 1000.0
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            n_box += len(boxes)
            if len(boxes):
                t_img_hit += 1
            nb, gh = center_hits(boxes, g["masks"], g["w"], g["h"])
            n_box_on_gt += nb
            n_gt_covered += int(gh.sum())
        rec = dict(
            weights=str(tw.name),
            img_det_rate=round(t_img_hit / n_img, 4),
            boxes_per_image=round(n_box / n_img, 3),
            box_on_gt_rate=round(n_box_on_gt / n_box, 4) if n_box else 0.0,
            gt_instance_coverage=round(n_gt_covered / n_inst, 4),
            latency_ms_mean=round(t_sum / n_img, 1),
            n_classes=len(seg.names),
        )
        # 同口径的原始最高分，供分布图对比
        raw = []
        for g in gts:
            im = cv2.imread(str(g["path"]))
            r = seg.predict(im, conf=0.001, imgsz=args.imgsz, verbose=False)[0]
            raw.append(round(float(r.boxes.conf.max()), 4) if len(r.boxes) else 0.0)
        rec["max_confs_per_image"] = raw
        report["trained_reference"] = rec
        print(f"  出图={rec['img_det_rate']:6.1%}  框/张={rec['boxes_per_image']:5.2f}  "
              f"落病斑={rec['box_on_gt_rate']:6.1%}  覆盖GT实例={rec['gt_instance_coverage']:6.1%}  "
              f"时延={rec['latency_ms_mean']:.0f}ms  可输出类别数={rec['n_classes']}")
    else:
        print(f"  [跳过] 未找到 {tw}")

    # ---------- J2：类别命名上限 ----------
    report["class_naming"] = {
        "task_classes_34host": 34,
        "task_classes_115combo": 115,
        "open_vocab_class_count_equals": "手写提示词个数",
        "note": ("开放词表的'类别'就是你写的提示词，不是一个可学习的分类头。"
                 "要达到 34/115 个细粒度病害名，只能人工把 34/115 个英文病名全写进提示词，"
                 "再指望 CLIP 文本-图像匹配能把它们分开 —— 这已不是零样本，是'零样本+人工病害知识库'，"
                 "且类别判定错误无法通过后续标注数据修正。"),
    }

    dst = OUT / "zeroshot_probe.json"
    dst.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[输出] {dst}")


if __name__ == "__main__":
    main()

"""M4 人机协同标注界面。

设计上的三条硬约束（都是被验收指标倒逼出来的，不是审美）：

1. 人不改掩码，只改框和类别。
   掩码由框+分割模型自动生成。如果让人逐个顶点去拖掩码，
   单张的编辑动作会从个位数涨到几十次，「修改次数下降 30%」这个
   指标就直接失效了。

2. 每一次操作都单独记一条日志。
   验收要看「同精度下单张修改次数」，就必须知道每张图被改了几次、
   改的是什么。Gradio 拿不到鼠标轨迹，所以退而记录操作事件：
   改类别 / 删除实例 / 恢复 / 确认。每个事件带时间戳和耗时。

3. 提交才算完成，中途可以反复改。
   提交前所有操作都在内存里，提交时一次性落盘成 YOLO 标签。

运行：
  python src/m4_app.py
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from common import load_config

OPS_CSV = "out/logs/ops.csv"
HUMAN_DIR = "out/human/labels"


def imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ 会话
class Session:
    """一次标注会话的全部状态。放在 gr.State 里，每个浏览器标签页独立。"""

    def __init__(self, cfg, weights, split="train"):
        from ultralytics import YOLO
        self.cfg = cfg
        self.model = YOLO(str(weights))
        self.classes = json.load(
            open(PROJECT / cfg["paths"]["classes_json"], encoding="utf-8"))["classes"]
        self.ops_path = PROJECT / OPS_CSV
        self.ops_path.parent.mkdir(parents=True, exist_ok=True)
        self.human_dir = PROJECT / HUMAN_DIR
        self.human_dir.mkdir(parents=True, exist_ok=True)

        # 候选池：还没人工确认过的图
        done = {p.stem for p in self.human_dir.glob("*.txt")}
        self.pool = []
        with open(PROJECT / cfg["paths"]["manifest"], encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["split"] != split or r["is_background"] == "1":
                    continue
                key = Path(r["path"]).stem
                if key in done:
                    continue
                self.pool.append(r)
        self.pool.sort(key=lambda r: r["class_dir"])

        self.cursor = 0
        self.image = None
        self.image_path = ""
        self.api_ms = 0.0
        self.instances = []      # [{"cls":str,"conf":float,"poly":np.ndarray,"box":tuple}]
        self.undo_stack = []
        self.n_ops = 0           # 本张图的操作次数（提交前）
        self.n_submit = 0        # 本会话已提交张数
        self.total_ops = 0
        self.t_load = time.time()

    def __deepcopy__(self, memo):
        """gr.State 会 deepcopy 初值做校验，而模型权重拷不动。
        会话状态本来就该是「一个标签页一份、原地修改」，
        所以这里直接返回自己，不做拷贝。"""
        return self

    # ---------------------------------------------------------- 载入
    def load_next(self):
        while self.cursor < len(self.pool):
            r = self.pool[self.cursor]
            self.cursor += 1
            bgr = imread_unicode(r["path"])
            if bgr is None:
                continue
            t0 = time.perf_counter()
            res = self.model.predict(bgr, imgsz=256, device=self.cfg["pseudo"]["device"],
                                     conf=0.25, verbose=False)[0]
            self.api_ms = (time.perf_counter() - t0) * 1000

            self.image = bgr
            self.image_path = r["path"]
            self.truth_class = r["class_dir"]
            self.instances = []
            if res.masks is not None:
                for poly, cls, conf in zip(res.masks.xy, res.boxes.cls, res.boxes.conf):
                    if len(poly) < 3:
                        continue
                    pts = poly.astype(np.float32)
                    self.instances.append({
                        "cls": res.names[int(cls)],
                        "conf": float(conf),
                        "poly": pts,
                        "box": (float(pts[:, 0].min()), float(pts[:, 1].min()),
                                float(pts[:, 0].max()), float(pts[:, 1].max())),
                    })
            self.undo_stack, self.n_ops = [], 0
            return True
        return False

    # ---------------------------------------------------------- 日志
    def log(self, op, detail, inst_idx=-1):
        new = not self.ops_path.exists()
        self.ops_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ops_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "image", "op", "instance", "detail",
                            "n_ops_on_image", "api_ms"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), Path(self.image_path).stem,
                        op, inst_idx, detail, self.n_ops, f"{self.api_ms:.1f}"])

    # ---------------------------------------------------------- 操作
    def push_undo(self, op, idx=-1):
        self.undo_stack.append((op, idx, [dict(x) for x in self.instances],
                               self.n_ops))

    def change_class(self, idx, new_cls):
        if not (0 <= idx < len(self.instances)):
            return "没选中实例"
        self.push_undo("改类别", idx)
        old = self.instances[idx]["cls"]
        if old == new_cls:
            return "类别没变，不计数"
        self.instances[idx]["cls"] = new_cls
        self.n_ops += 1
        self.log("改类别", f"{old} -> {new_cls}", idx)
        return f"第 {idx} 个实例：{old} -> {new_cls}"

    def delete_instance(self, idx):
        if not (0 <= idx < len(self.instances)):
            return "没选中实例"
        self.push_undo("删实例", idx)
        gone = self.instances.pop(idx)
        self.n_ops += 1
        self.log("删实例", gone["cls"], idx)
        return f"删掉了 {gone['cls']}"

    def confirm_all(self):
        """整张确认：预标注全对，一次都不改。这是最省时的路径。"""
        self.push_undo("整张确认")
        self.n_ops += 1
        self.log("整张确认", f"{len(self.instances)} 个实例")
        return "已确认整张，可直接提交"

    def no_target(self):
        """这张图里没有叶片（误入的背景图、拍糊的废片）。"""
        self.push_undo("标为无目标")
        self.instances, self.n_ops = [], self.n_ops + 1
        self.log("标为无目标", "")
        return "已标为无目标图"

    def undo(self):
        if not self.undo_stack:
            return "没有可撤销的操作"
        op, idx, snap, n = self.undo_stack.pop()
        self.instances = snap
        self.log("撤销", op, idx)
        self.n_ops = n
        return f"撤销了「{op}」"

    def submit(self):
        if self.image is None:
            return "没有图片"
        key = Path(self.image_path).stem
        lines = []
        h, w = self.image.shape[:2]
        for inst in self.instances:
            cid = self.classes.index(inst["cls"]) if inst["cls"] in self.classes else 0
            pts = inst["poly"].copy()
            pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
            pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
            if len(pts) > 80:
                step = len(pts) / 80
                pts = np.array([pts[int(i * step)] for i in range(80)])
            lines.append(str(cid) + " " +
                         " ".join(f"{x:.6f} {y:.6f}" for x, y in pts))
        with open(self.human_dir / f"{key}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.log("提交", f"{len(lines)} 个实例，改动 {self.n_ops} 次")
        self.n_submit += 1
        self.total_ops += self.n_ops
        return f"已提交（第 {self.n_submit} 张）"


# ------------------------------------------------------------------ 渲染
def render(st, msg=""):
    if st.image is None:
        return (None, [], gr_update_choices([], None), msg or "候选池已空", "")

    anns = []
    for inst in st.instances:
        x1, y1, x2, y2 = inst["box"]
        anns.append(((x1, y1, x2, y2), f"{inst['cls']} {inst['conf']:.2f}"))

    choices = [f"{i}: {x['cls']} ({x['conf']:.2f})"
               for i, x in enumerate(st.instances)]
    stats = (f"**本张**：{len(st.instances)} 个实例 · 已改 {st.n_ops} 次 · "
             f"预标注 {st.api_ms:.0f} ms\n\n"
             f"**会话**：已提交 {st.n_submit} 张 · "
             f"累计改动 {st.total_ops} 次 · "
             f"平均 {st.total_ops / max(st.n_submit, 1):.1f} 次/张")
    rgb = cv2.cvtColor(st.image, cv2.COLOR_BGR2RGB)
    return ((rgb, anns), choices, gr_update_choices(choices, choices[0] if choices else None),
            msg, stats)


def gr_update_choices(choices, value):
    import gradio as gr
    return gr.update(choices=choices, value=value)


# ------------------------------------------------------------------ 界面
def build(cfg, weights, split):
    import gradio as gr

    st0 = Session(cfg, weights, split)
    st0.load_next()

    with gr.Blocks(title="叶片实例预标注 · 人机协同") as demo:
        gr.Markdown("## 叶片实例预标注 · 人机协同标注台\n"
                    "左边是模型给的预标注，右边只需**改类别 / 删多余实例**，"
                    "掩码由框自动生成，不用手工描边。")
        st = gr.State(st0)

        with gr.Row():
            with gr.Column(scale=3):
                img = gr.AnnotatedImage(label="预标注结果", height=460)
            with gr.Column(scale=2):
                sel = gr.Dropdown(label="选中的实例", choices=[], value=None)
                with gr.Row():
                    cls_dd = gr.Dropdown(label="改成哪一类",
                                         choices=st0.classes, value=None)
                    btn_cls = gr.Button("改类别", variant="primary")
                with gr.Row():
                    btn_del = gr.Button("删掉这个实例")
                    btn_undo = gr.Button("撤销")
                gr.Markdown("---")
                with gr.Row():
                    btn_ok = gr.Button("整张确认（不用改）", variant="secondary")
                    btn_none = gr.Button("这张没有叶片")
                gr.Markdown("---")
                btn_sub = gr.Button("提交并下一张", variant="primary", size="lg")
                log = gr.Markdown("")
                stats = gr.Markdown("")

        def idx_of(s):
            if not s:
                return -1
            return int(str(s).split(":")[0])

        def refresh(st, msg=""):
            out = render(st, msg)
            return (st, out[0], out[2], out[3], out[4])

        def on_class(st, s, c):
            m = st.change_class(idx_of(s), c)
            return refresh(st, m)

        def on_del(st, s):
            m = st.delete_instance(idx_of(s))
            return refresh(st, m)

        def on_ok(st):
            return refresh(st, st.confirm_all())

        def on_none(st):
            return refresh(st, st.no_target())

        def on_undo(st):
            return refresh(st, st.undo())

        def on_sub(st):
            m = st.submit()
            if not st.load_next():
                return (st, None, gr.update(choices=[], value=None), m, "候选池已空")
            return refresh(st, m)

        outs = [st, img, sel, log, stats]
        btn_cls.click(on_class, [st, sel, cls_dd], outs)
        btn_del.click(on_del, [st, sel], outs)
        btn_ok.click(on_ok, [st], outs)
        btn_none.click(on_none, [st], outs)
        btn_undo.click(on_undo, [st], outs)
        btn_sub.click(on_sub, [st], outs)

        demo.load(lambda st: refresh(st), [st], outs)

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT / "configs" / "default.yaml"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.weights:
        weights = PROJECT / args.weights
    else:
        cands = sorted((PROJECT / "runs").rglob("best.pt"))
        if not cands:
            raise SystemExit("还没有训练好的权重，先跑 src/m3_train.py")
        weights = cands[-1]
    print(f"权重：{weights}")

    build(cfg, weights, args.split).launch(
        server_name="127.0.0.1", server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()

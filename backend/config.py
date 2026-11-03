"""后端配置：路径解析、数据源判定、设备选择。

约定：换机器只改环境变量，不改代码。
    AGROLABEL_DATA_ROOT   数据集根目录（缺省读 configs 里的 path）
    AGROLABEL_WEIGHTS     权重路径
    AGROLABEL_DEVICE      推理设备，缺省 "0"（GPU0），无卡时自动退 cpu
    AGROLABEL_CONF        默认置信度阈值
    AGROLABEL_IMGSZ       推理尺寸
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 类别字典

def _load_classes() -> tuple[list[str], int]:
    """类别名的唯一事实来源：configs/dataset_plantseg34.local.yaml 的 names。"""
    cfg = PROJECT / "configs" / "dataset_plantseg34.local.yaml"
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    nc = int(raw["nc"])
    names = raw["names"]
    # YAML 的 key 是 int，但保险起见兼容 str
    table = {int(k): str(v) for k, v in names.items()}
    missing = [i for i in range(nc) if i not in table]
    if missing:
        raise RuntimeError(f"classes 缺编号 {missing}，检查 {cfg}")
    return [table[i] for i in range(nc)], nc


CLASS_NAMES, NC = _load_classes()

# 类别名 -> id 的反查表，predict 时做名字规范化用
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------- 数据源

@dataclass(frozen=True)
class DataSource:
    mode: str              # "manifest" | "yolo" | "demo"
    root: Path | None      # manifest 模式下的数据集根
    demo_dir: Path | None  # demo 模式下的图片目录
    note: str


def _looks_like_yolo(root: Path) -> bool:
    """YOLO-seg 目录结构：images/ 下有 split 子目录，且带 data.yaml。"""
    return (root / "images").is_dir() and (root / "data.yaml").is_file()


def resolve_data_source() -> DataSource:
    """优先真实数据集；没有就退到仓库内可用的样例图。

    这样队友拉下代码不配数据集也能把前端跑起来，不至于卡在 F:/_plantseg。
    """
    env_root = os.environ.get("AGROLABEL_DATA_ROOT")
    if env_root:
        p = Path(env_root)
        if not p.is_dir():
            raise RuntimeError(f"AGROLABEL_DATA_ROOT 指向的目录不存在: {p}")
        # 自动识别：有 images/ + data.yaml 就是 YOLO 目录结构，否则按 COCO 清单处理
        mode = "yolo" if _looks_like_yolo(p) else "manifest"
        return DataSource(mode, p, None, f"由环境变量指定: {p}")

    cfg = PROJECT / "configs" / "dataset_plantseg34.local.yaml"
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    declared = Path(raw["path"])
    if declared.is_dir():
        mode = "yolo" if _looks_like_yolo(declared) else "manifest"
        return DataSource(mode, declared, None, f"configs 声明且存在: {declared}")

    # 本地采样包（datasets/plantseg100_pack，100 张 × 34 类，YOLO-seg 格式）。
    # 它比 demo 更像真数据，所以排在 demo 前面。
    pack = PROJECT / "datasets" / "plantseg100_pack"
    if (pack / "images").is_dir() and (pack / "data.yaml").is_file():
        n = sum(1 for s in (pack / "images").iterdir() if s.is_dir()
                for f in s.iterdir() if f.suffix.lower() in (".jpg", ".png"))
        return DataSource(
            "yolo", pack, None,
            f"本地采样包: datasets/plantseg100_pack（{n} 张，34 类，含 al_pool/ 待标注池）",
        )

    # demo 首选：由 tools/build_demo_set.py 从评测双联图裁出的**真实病斑原图**
    demo_built = PROJECT / "out" / "demo"
    if (demo_built / "_manifest.json").is_file():
        return DataSource(
            "demo", None, demo_built,
            f"{declared} 不存在；使用 out/demo/（{len(list(demo_built.glob('*.jpg')))} 张真实病斑图，"
            f"由 build_demo_set.py 生成，仅供联调）",
        )

    # demo 兜底：仓库自带的评测可视化图。注意这些是**双联对比图**（原图｜标注叠加），
    # 不是可直接标注的原始照片，只能用来确认界面能跑起来。
    # 上游已把 sam3_sample_vis/ 删除并 gitignore，这里改指向仍然入库的 conversion_preview。
    fallback = PROJECT / "experiments" / "sam3_conversion_preview"
    if fallback.is_dir() and any(fallback.glob("*.jpg")):
        return DataSource(
            "demo", None, fallback,
            f"{declared} 不存在，out/demo/ 也没生成；退用 {fallback.name}/ 的评测对比图（双联拼接，"
            f"仅够确认界面能启动）。要正常样例图请先跑 backend/tools/build_demo_set.py",
        )

    raise RuntimeError(
        f"找不到任何图像数据：{declared} 不存在，{demo_built} 与 {fallback} 都为空。\n"
        f"请设置 AGROLABEL_DATA_ROOT 指向 PlantSeg 数据集根目录，"
        f"或先运行 backend/tools/build_demo_set.py 生成样例图。"
    )


DATA = resolve_data_source()


# ---------------------------------------------------------------- 模型

WEIGHTS = Path(os.environ.get(
    "AGROLABEL_WEIGHTS",
    str(PROJECT / "weights" / "baseline_yolov8nseg_plantseg34_best.pt"),
))

CONF = float(os.environ.get("AGROLABEL_CONF", "0.25"))
IMGSZ = int(os.environ.get("AGROLABEL_IMGSZ", "640"))


def resolve_device() -> str:
    """返回 ultralytics 认的 device 字符串。请求 GPU 但没卡时降级为 cpu。"""
    want = os.environ.get("AGROLABEL_DEVICE", "0")
    if want in ("cpu", ""):
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return want
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------- 输出目录

OUT_DIR = PROJECT / "out"
PRED_CACHE_DIR = OUT_DIR / "pred_cache"
ANNOTATION_DIR = OUT_DIR / "annotations"
LOG_DIR = OUT_DIR / "logs"
OPS_CSV = LOG_DIR / "ops.csv"

# 实验臂：标注产物落到 out/annotations/<arm>/，三臂互不覆盖（ADDENDUM §4.3）
# assisted = 预标 + 修正 + 点提示，manual = 纯人工对照，noprompt = 预标但禁用点提示
DEFAULT_ARM = os.environ.get("AGROLABEL_ARM", "assisted")

# 白名单：臂名拼错一个字母就会静默分裂出一个新目录，三臂对照直接作废。
# 所以未知臂名一律 422，宁可报错也别让脏目录产生。
VALID_ARMS = ("assisted", "manual", "noprompt")


def ensure_dirs() -> None:
    for d in (OUT_DIR, PRED_CACHE_DIR, ANNOTATION_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

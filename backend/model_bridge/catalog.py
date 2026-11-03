"""图像索引：把磁盘上的图整理成 (image_id -> 记录) 的只读目录。

三种来源：
  manifest —— 读 datasets/sam3/plantseg34_{train,val}.json（COCO），路径拼到 DATA_ROOT 上，
              只保留磁盘上真实存在的条目。宽高取自清单，不解码原图。
  yolo     —— 直接扫 YOLO-seg 目录结构：images/<split>/*.jpg + labels/<split>/*.txt
              + data.yaml。采样包 datasets/plantseg100_pack 就是这种。
  demo     —— 读 out/demo/_manifest.json（由 backend/tools/build_demo_set.py 生成），
              没有清单时退回扫目录、用 PIL 现读尺寸。

⚠️ 这里刻意不用 cv2.imread：OpenCV 在 Windows 非 ASCII 路径下会读失败。
本仓库路径含中文，ultralytics 导入后会给 cv2 打补丁，但那是隐式依赖，
不该让索引逻辑依赖"某个模块先被导入过"。PIL 没有这个问题。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
import yaml

from .. import config


@dataclass
class ImageRec:
    image_id: str
    path: Path
    width: int
    height: int
    split: str
    gt_class_id: int | None = None
    gt_class_name: str | None = None
    disease: str | None = None
    # 排序用：manifest 里保留原始顺序，demo 里按文件名
    order: int = field(default=0)

    def public(self) -> dict:
        return {
            "image_id": self.image_id,
            "file_name": self.path.name,
            "split": self.split,
            "width": self.width,
            "height": self.height,
            "gt_class_id": self.gt_class_id,
            "gt_class_name": self.gt_class_name,
            "disease": self.disease,
        }

    def posix_key(self) -> str:
        return f"{self.split}/{self.path.name}"


class Catalog:
    def __init__(self) -> None:
        self._by_id: dict[str, ImageRec] = {}
        self._ordered: list[ImageRec] = []
        self.mode = "empty"
        self.skipped = 0

    # ------------------------------------------------------------ 构建

    def build(self) -> None:
        if config.DATA.mode == "manifest":
            self._build_manifest(config.DATA.root)
        elif config.DATA.mode == "yolo":
            self._build_yolo(config.DATA.root)
        elif config.DATA.mode == "demo":
            self._build_demo(config.DATA.demo_dir)
        if not self._ordered:
            raise RuntimeError(f"索引为空（mode={config.DATA.mode}）。检查 {config.DATA.note}")

    def _add(self, rec: ImageRec) -> None:
        if rec.image_id in self._by_id:
            return
        self._by_id[rec.image_id] = rec
        self._ordered.append(rec)

    def _build_manifest(self, root: Path | None) -> None:
        assert root is not None
        self.mode = "manifest"
        seq = 0
        for fname in ("plantseg34_train.json", "plantseg34_val.json"):
            p = config.PROJECT / "datasets" / "sam3" / fname
            if not p.is_file():
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            # categories: id -> name（COCO 的 id 从 1 开始，我们的 class_id 从 0）
            cats = {c["id"]: c["name"] for c in data.get("categories", [])}
            for img in data.get("images", []):
                rel = img.get("image_path") or img.get("file_name")
                abs_path = root / rel
                if not abs_path.is_file():
                    self.skipped += 1
                    continue
                cls_name = img.get("host") or cats.get(img.get("category_id"), "")
                cls_id = config.CLASS_TO_ID.get(cls_name)
                self._add(ImageRec(
                    image_id=str(img["id"]),
                    path=abs_path,
                    width=int(img.get("width") or 0),
                    height=int(img.get("height") or 0),
                    split=str(img.get("split") or "train"),
                    gt_class_id=cls_id,
                    gt_class_name=cls_name or None,
                    disease=img.get("disease"),
                    order=seq,
                ))
                seq += 1

    _DEMO_RE = re.compile(r"^(\d+)_(.+?)_(\d+)$")

    # ---------------------------------------------------------------- yolo 模式

    _IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    @staticmethod
    def _yaml_names(root: Path) -> list[str]:
        """从 data.yaml 读类别表。包内的这份是权威来源，优先于全局 config。"""
        p = root / "data.yaml"
        if not p.is_file():
            return []
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return []
        names = raw.get("names")
        if isinstance(names, dict):           # {0: 'Apple', 1: 'Banana', ...}
            return [str(names[k]) for k in sorted(names, key=int)]
        if isinstance(names, list):
            return [str(x) for x in names]
        return []

    @staticmethod
    def _first_class_id(label_path: Path) -> int | None:
        """YOLO-seg 每行 `class_id x1 y1 ...`：取第一行第一个字段当这张图的 gt 类。

        一张图上可能有多个类（同一宿主下多种病害），但面板只显示一个"这张图属于哪类"，
        取首个实例足够，不猜"多数类"——那需要读完整个文件，收益很小。
        """
        if not label_path.is_file():
            return None
        try:
            with label_path.open(encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split()
                    if parts:
                        return int(parts[0])
        except (OSError, ValueError):
            return None
        return None

    @staticmethod
    def _size(path: Path) -> tuple[int, int]:
        try:
            with Image.open(path) as im:
                return im.width, im.height
        except Exception:                      # noqa: BLE001  坏图不该让建索引整个失败
            return 0, 0

    def _build_yolo(self, root: Path | None) -> None:
        """扫 YOLO-seg 目录：images/<split>/* 配 labels/<split>/<stem>.txt。"""
        assert root is not None
        self.mode = "yolo"

        names = self._yaml_names(root) or list(config.CLASS_NAMES)

        seq = 0
        for split_dir in sorted(p for p in (root / "images").iterdir() if p.is_dir()):
            split = split_dir.name
            for img in sorted(split_dir.iterdir()):
                if img.suffix.lower() not in self._IMG_EXT:
                    continue
                cid = self._first_class_id(
                    root / "labels" / split / f"{img.stem}.txt"
                )
                w, h = self._size(img)
                if w == 0 or h == 0:
                    self.skipped += 1
                    continue

                # image_id 会成为标注文件名（out/annotations/<arm>/<id>.txt），
                # 不能带路径分隔符；跨 split 重名时才加前缀
                image_id = img.stem
                if image_id in self._by_id:
                    image_id = f"{split}_{img.stem}"

                self._add(ImageRec(
                    image_id=image_id,
                    path=img,
                    width=w,
                    height=h,
                    split=split,
                    gt_class_id=cid,
                    gt_class_name=(
                        names[cid] if cid is not None and 0 <= cid < len(names) else None
                    ),
                    disease=None,
                    order=seq,
                ))
                seq += 1

    def _build_demo(self, demo_dir: Path | None) -> None:
        assert demo_dir is not None
        self.mode = "demo"

        # 有 _manifest.json 就直接读，省掉逐张解码
        mf = demo_dir / "_manifest.json"
        if mf.is_file():
            data = json.loads(mf.read_text(encoding="utf-8"))
            for i, it in enumerate(data.get("items", [])):
                p = demo_dir / it["file"]
                if not p.is_file():
                    self.skipped += 1
                    continue
                self._add(ImageRec(
                    image_id=p.stem,
                    path=p,
                    width=int(it.get("width") or 0),
                    height=int(it.get("height") or 0),
                    split="demo",
                    gt_class_id=it.get("class_id"),
                    gt_class_name=it.get("class_name"),
                    disease=it.get("disease"),
                    order=i,
                ))
            return

        # 没有清单就自己扫；尺寸用 PIL 读 ——
        # 不要用 cv2.imread，它在 Windows 非 ASCII 路径下会失败（除非 ultralytics 已导入打了补丁）
        from PIL import Image

        for i, p in enumerate(sorted(demo_dir.glob("*.jpg"))):
            m = self._DEMO_RE.match(p.stem)
            cls_name = m.group(2).replace("_", " ") if m else None
            w = h = 0
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                pass
            self._add(ImageRec(
                image_id=p.stem,
                path=p,
                width=int(w),
                height=int(h),
                split="demo",
                gt_class_id=config.CLASS_TO_ID.get(cls_name) if cls_name else None,
                gt_class_name=cls_name,
                disease=None,
                order=i,
            ))

    # ------------------------------------------------------------ 查询

    def get(self, image_id: str) -> ImageRec | None:
        return self._by_id.get(image_id)

    def __len__(self) -> int:
        return len(self._ordered)

    def query(
        self,
        *,
        split: str | None = None,
        class_id: int | None = None,
        ids: list[str] | None = None,
        offset: int = 0,
        limit: int = 50,
        order: str = "path",
    ) -> tuple[list[ImageRec], int]:
        rows = self._ordered

        if ids:
            wanted = set(ids)
            rows = [r for r in rows if r.image_id in wanted]
            # ids 模式下按调用方给的顺序返回，最多 200 个
            pos = {v: i for i, v in enumerate(ids)}
            rows = sorted(rows, key=lambda r: pos.get(r.image_id, 1 << 30))
            return rows[:200], len(rows)

        if split and split != "all":
            rows = [r for r in rows if r.split == split]
        if class_id is not None:
            rows = [r for r in rows if r.gt_class_id == class_id]

        total = len(rows)

        if order == "path":
            rows = sorted(rows, key=lambda r: (r.posix_key(), r.order))
        elif order == "class":
            rows = sorted(rows, key=lambda r: (r.gt_class_id if r.gt_class_id is not None else 999, r.path.name))
        # 其余 order 目前无意义（score_desc 要等 pred 缓存齐了才有数据），退回原顺序

        return rows[offset: offset + limit], total

    def splits(self) -> list[dict]:
        from collections import Counter
        c = Counter(r.split for r in self._ordered)
        return [{"name": k, "count": v} for k, v in sorted(c.items())]

    def class_counts(self) -> list[dict]:
        from collections import Counter
        c = Counter(r.gt_class_id for r in self._ordered)
        out = []
        for idx, name in enumerate(config.CLASS_NAMES):
            n = c.get(idx, 0)
            if n:
                out.append({"class_id": idx, "class_name": name, "count": n})
        return out

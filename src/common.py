"""公共函数：配置加载、图片遍历、类别解析、可视化。

只放被两个以上脚本用到的函数。只有一处用的，留在各自脚本里。
"""
import os
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config(path="configs/default.yaml"):
    """读配置。只用 PyYAML 一个额外依赖。"""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(root: Path, *parts):
    """把配置里的相对路径接到项目根目录上。"""
    return root / Path(*parts)


def iter_images(root):
    """遍历目录下所有图片。

    不用 glob('*.jpg')，原因是实测这份数据有四种扩展名
    （.JPG / .jpg / .jpeg / .png），大小写混用。
    这里显式后缀集合 + lower()，一次解决。
    """
    root = Path(root)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                yield Path(dirpath) / fn


def parse_class_name(dirname: str, sep: str = "___"):
    """'Tomato___Early_blight' -> ('Tomato', 'Early blight')

    注意 Spider_mites 那个目录名里有空格，不能用 split() 切分。
    """
    if sep in dirname:
        crop, disease = dirname.split(sep, 1)
    else:
        crop, disease = dirname, ""
    return crop.strip(), disease.strip().replace("_", " ")


def human_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def setup_matplotlib():
    """让 matplotlib 能显示中文。Windows 上 YaHei 一定在。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt

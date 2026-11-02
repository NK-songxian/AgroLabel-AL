#!/usr/bin/env bash
# AutoDL Pro6000 一次性环境准备（幂等）
#
# 约定：项目相关的所有东西都放在仓库目录内，仓库之外不落任何文件。
#
#   <repo>/data/plantseg34     数据集（不入 git）
#   <repo>/weights/sam3.pt     SAM3 原始权重（不入 git）
#   <repo>/third_party/sam3_repo  SAM3 官方源码（不入 git）
#   <repo>/envs/sam3train      隔离 venv（不入 git，继承 conda base 的 torch）
#   <repo>/runtime/            运行产物（不入 git）
#
# 不改动全局 python、不杀进程、不写仓库之外的路径。
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENVP="$REPO/envs/sam3train"
BASE_PY="${BASE_PY:-/root/miniconda3/bin/python}"
CKPT="$REPO/weights/sam3.pt"
DATA="$REPO/data/plantseg34"
SAM3_SRC="$REPO/third_party/sam3_repo"

cd "$REPO"

echo "== 1. GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo "== 2. 隔离 venv（--system-site-packages 继承 base 的 torch，不重复装 2.5G）=="
if [ ! -x "$ENVP/bin/python" ]; then
  "$BASE_PY" -m venv --system-site-packages "$ENVP"
fi
PY="$ENVP/bin/python"
"$PY" -V
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== 3. SAM3 源码（可编辑安装，指向 third_party/sam3_repo）=="
if [ ! -d "$SAM3_SRC/sam3" ]; then
  echo "缺少 $SAM3_SRC，请先解压/克隆 SAM3 官方源码" >&2
  exit 1
fi
"$PY" -m pip install -q --no-deps -e "$SAM3_SRC"
"$PY" -c "import sam3, os; print('sam3 ->', os.path.dirname(sam3.__file__))"

echo "== 4. 评测依赖 =="
"$PY" -m pip install -q pycocotools pillow numpy opencv-python-headless

echo "== 5. 检查权重 =="
if [ -f "$CKPT" ]; then
  ls -lh "$CKPT"
else
  echo "警告：$CKPT 不存在，需先放好 sam3.pt（3.3G）"
fi

echo "== 6. 检查数据 =="
if [ -d "$DATA/images/val" ]; then
  # images/ 下混放了同名 .npy 掩码，只数图片，别把掩码算成图
  n_val=$(find "$DATA/images/val" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
  n_train=$(find "$DATA/images/train" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | wc -l)
  echo "数据 OK: train $n_train / val $n_val"
else
  echo "警告：$DATA 不存在，先上传数据"
fi

echo "== 完成 =="
echo "解释器: $PY"

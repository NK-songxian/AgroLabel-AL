#!/usr/bin/env bash
# AutoDL Pro6000 一次性环境准备（幂等，只写 /root/autodl-tmp/Ryan/）
# 不改动全局 python、不杀进程、不动 Ryan/ 之外的任何文件。
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Ryan}"
REPO="$ROOT/ArgoLable-AL"
ENVP="$ROOT/envs/sam3"
CKPT="$ROOT/sam3.pt"
DATA="$ROOT/data_plantseg34"

echo "== 0. 目录 =="
mkdir -p "$ROOT/envs" "$ROOT/runs" "$ROOT/logs"
if [ ! -d "$REPO/.git" ]; then
  echo "克隆仓库到 $REPO"
  rmdir "$REPO" 2>/dev/null || true
  git clone https://github.com/NK-songxian/AgroLabel-AL.git "$REPO"
fi
cd "$REPO" && git pull -q --ff-only || true

echo "== 1. GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo "== 2. conda 环境（独立前缀，不污染 base）=="
if [ ! -x "$ENVP/bin/python" ]; then
  conda create -y -p "$ENVP" python=3.12
fi
PY="$ENVP/bin/python"
"$PY" -V

echo "== 3. PyTorch (cu128) =="
if ! "$PY" -c "import torch" 2>/dev/null; then
  "$PY" -m pip install -q torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
fi
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "== 4. SAM3 源码 =="
if [ ! -d "$REPO/third_party/sam3" ]; then
  mkdir -p "$REPO/third_party"
  git clone --depth 1 https://github.com/facebookresearch/sam3.git "$REPO/third_party/sam3"
fi
"$PY" -m pip install -q -e "$REPO/third_party/sam3"

echo "== 5. 评测依赖 =="
"$PY" -m pip install -q pycocotools pillow numpy

echo "== 6. 检查权重 =="
if [ -f "$CKPT" ]; then
  ls -lh "$CKPT"
else
  echo "警告：$CKPT 不存在，需先放好 sam3.pt"
fi

echo "== 7. 检查数据 =="
if [ -d "$DATA/images/val" ]; then
  echo "数据 OK: $(ls "$DATA/images/val" | wc -l) val 张"
  [ -e "$REPO/data/plantseg34" ] || ln -sfn "$DATA" "$REPO/data/plantseg34"
else
  echo "警告：$DATA 不存在，先上传数据"
fi

echo "== 完成 =="
echo "激活用: $ENVP/bin/python  （或 conda activate $ENVP）"

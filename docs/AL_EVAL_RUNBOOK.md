# AL 效率消融 · 队友运行说明（v1.1，2026-11-17）

## 0. 你收到什么

- `AgroLabel_AL_full_dataset_v3.zip`：7773 张 JPEG（train 5367 / val 846 / test 1560）+ YOLO labels + data.yaml/classes.json + `m8_al.py` + 本说明 + **官方 COCO 冷启动权重 `weights/yolo26x-seg.pt`**
- `AgroLabel_B6_handoff.zip`：B6 最优权重（mask mAP50 0.416）+ 前后端启动材料（**仅用于演示/人工编辑对照，不用于 AL 训练**）

**注意**：原数据目录里的 `.npy` 是 JPEG 的 uint8 像素副本（13G），AL 完全不用，已从包中剔除。

## 1. 环境

```bash
conda create -n al_run python=3.11 -y
conda activate al_run
pip install ultralytics opencv-python pyyaml numpy pillow scipy
# torch 需 CUDA 版（有 GPU 时）：pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

## 2. 文件摆放

```text
<你的工作目录>/
├── dataset/            ← 解压 AgroLabel_AL_full_dataset_v3.zip（内含 images/ labels/ data.yaml）
├── weights/yolo26x-seg.pt               ← **官方 COCO 预训练权重**（AL 起点，v3 包内已提供）
└── m8_al.py
```

如果你拿到的是旧的 `v2` 包（里面没有 `weights/yolo26x-seg.pt`）：首次运行 `--model yolo26x-seg.pt` 时 Ultralytics 会自动联网下载（约 136MB）；无网络则必须先把官方权重放进 `weights/` 再跑。v3 包已内置，无需这一步。

**重要**：`data.yaml` 里的 `path:` 是强哥本机的绝对路径，解压后要改成你自己的 `dataset` 路径（只改这一行）：
```yaml
path: <你的绝对路径>/dataset
train: images/train
val: images/val
test: images/test
nc: 34
names: ...（不要动）
```

## 3. 跑 AL 消融（核心命令）

```bash
python m8_al.py \
  --data dataset/data.yaml \
  --strategies random entropy tta coreset \
  --rounds 5 \
  --init-per-class 3 \
  --budget-step 0.08 \
  --model weights/yolo26x-seg.pt \
  --epochs 25 --imgsz 1280 --batch 8 --device 0 \
  --cand-mult 5 \
  --work-root ./al_runs --name al_b6_from_coco
```

## ⚠️ 关键警告：AL 起点**绝不能**用 B6 权重

`m8_al.py` 第 353 行 `weights = args.model`——第一轮训练直接以传入权重为起点。B6（`b6_y26x_1280_best.pt`）是用 **全部 5367 张 train 图**训练的，若用它作 AL 起点：

1. 模型在 r0 之前就已“见过”所有训练图 → 不确定性采样失去意义；
2. test mAP 起点虚高、曲线平坦 → **AL 效率对比彻底失真**，比赛证据作废；
3. 这与“把训练数据泄露给评测”等价，属于方法学错误。

**正确做法**：用官方 COCO 预训练 `yolo26x-seg.pt`（没见过 PlantSeg 任何图）冷启动，每轮从上一轮 `last.pt` 继续——这是标准 pool-based AL 协议。B6 权重仅用于：前后端预标注演示 / 人工编辑效率对照的预标注底稿，**不参与 AL 训练循环**。

**口径说明**：
- 初始标注：每类 3 张（≈102 张），每轮增 8% 未标注池（≈420 张），5 轮后每策略 ≈2200 张
- 4 策略共享**同一批候选子集**（cand-mult=5），保证公平对比
- 每轮训练 25 epochs 固定，不挑 best（val=False），最后统一在 test split 评估
- 单策略约 2–3h（GPU），4 策略合计 10–12h；无 GPU 请用 `--device cpu --epochs 5 --rounds 2` 冒烟

## 4. 已有冒烟结果（你复现的对照锚点）

> 此冒烟是在云端用**官方 yolo26x-seg.pt 冷启动**跑的（不是 B6），与上面的正确协议一致。

| 策略 | 轮次 | 已标注 | mask mAP50 |
|---|---|---|---|
| random（冒烟，2 轮，15ep） | r0 | 102 | 0.0154 |
| | r1 | 523 | 0.0604 |
| | r2 | 944 | 0.1094 |

全监督参考线：B5 0.397 / **B6 0.416**（5367 张全量训练）。

## 5. 回传物

1. `al_runs/` 目录（每策略一个 csv/json 结果，脚本自动写）
2. `results.json`（策略 × 轮次 × mAP 曲线，画对比图用）
3. 一句话结论：同预算下哪个策略 mAP 最高，差距多少

## 6. 已知坑（已验证）

- `m8_al.py` 已做独立包适配：脚本在仓库 `src/` 下时输出到仓库根，解压在单独目录时输出到该解压目录内的 `runs/` 和 `out/`，不再写到上一级；
- YOLO 训练时 `cache=False` 已设，别开缓存（数据 16G 会撑爆内存）；
- 强哥云端实测冒烟：102→0.015、523→0.060、944→0.109，若你的 r0 结果差一个数量级，先检查 data.yaml 的 names 顺序是否被改动。

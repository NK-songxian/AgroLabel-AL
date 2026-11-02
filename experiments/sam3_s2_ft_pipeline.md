# SAM3 微调管线打通记录（S2 全量微调）

> 目标：把官方 `sam3/train/train.py`（Hydra + submitit）接到 PlantSeg-34 的
> 病斑分割数据上，跑通 S2 全量微调。本文件记录**踩到的每一个坑和修法**，
> 便于 S3（LoRA）/ S4（病害级提示）复用。

## 1. 结论先行

| 项 | 结论 |
| --- | --- |
| 数据接入方式 | 不需要重排目录。`Sam3ImageDataset` 的 `img_folder` / `ann_file` 是两个独立字段，直接喂 `datasets/sam3/plantseg34_{train,val}.json` 即可 |
| 训练提示词 | 来自 COCO `categories[].name`；用 `coco_json_loader.prompts`（JSON 字符串）覆盖，格式 `[{"id":1,"name":"...","supercategory":"..."}]`，条数必须与类别数一致 |
| 分割必须开三个开关 | `scratch.enable_segmentation: True`、`loss_fns_find` 里加 `Masks`、collator `with_seg_masks: True` |
| 单步耗时 | 1008²、1 图/步、每图 34 个文本查询：**1.24 s/步**，显存 **40–44 GB**（RTX PRO 6000 96G） |
| 一个 epoch | 5367 步 ≈ **2.2 h** |

## 2. 环境（隔离，不动 base）

```bash
/root/miniconda3/bin/python -m venv --system-site-packages /root/autodl-tmp/Ryan/envs/sam3train
P=/root/autodl-tmp/Ryan/envs/sam3train/bin/pip
$P install --no-deps -e /root/autodl-tmp/Ryan/sam3_repo
$P install hydra-core submitit tensorboard zstandard torchmetrics fvcore fairscale scikit-image scikit-learn iopath
```

- 用 `--system-site-packages` 继承 base 的 torch 2.8.0+cu128 / numpy 2.3.2 / ultralytics，
  避免 `pip install -e .` 把 numpy 降到 <2 而影响 base 上别人的进程（pip 只会提示冲突，不实际降级）。
- `--no-deps` 是必须的：sam3 声明 `numpy<2`、`ftfy==6.1.1`，直接装会改坏共享环境。

## 3. 配置生成

```bash
python datasets_tools/sam3_make_ft_config.py \
    --exp-name s2_full_ft_host --max-epochs 10 --val-freq 2 --val-images 150 --chunk 34 \
    --out configs/sam3/plantseg34_s2.yaml
cp configs/sam3/plantseg34_s2.yaml /root/autodl-tmp/Ryan/sam3_repo/sam3/train/configs/plantseg34/
```

**yaml 必须放进 `sam3/train/configs/` 下**：`train.py` 用
`initialize_config_module("sam3.train")`，`-c` 的 `config_name` 是相对该模块目录解析的，
绝对路径不会被接受。

启动：

```bash
cd /root/autodl-tmp/Ryan/sam3_repo
nohup /root/autodl-tmp/Ryan/envs/sam3train/bin/python sam3/train/train.py \
    -c configs/plantseg34/plantseg34_s2.yaml --use-cluster 0 --num-gpus 1 \
    > /root/autodl-tmp/Ryan/logs/sam3_s2_full.log 2>&1 &
```

## 4. 六个坑（按踩到顺序）

### 坑 1：官方 tarball 里没有 `configs/`

GitHub 仓库的 `configs/` 实际在 `sam3/train/configs/`（不在仓库根）。
`ls configs/` 会报不存在，别以为没发配置。

### 坑 2：`gradient_accumulation_steps > 1` 直接断言失败

`trainer._run_step` 在 `grad_accum > 1` 时要求 dataloader **直接吐出 `list[N]`**，
但 `TorchDataset`/`DataLoader` 只会给单个 batch dict：

```
AssertionError: Expected a list of batches, got <class 'dict'>
```

官方 recipe 一律 `gradient_accumulation_steps: 1`，跟随即可。

### 坑 3：trunk MLP 是**只支持推理**的融合算子

`sam3/model/vitdet.py::Mlp.forward` 直接调 `sam3/perflib/fused.addmm_act`，而它内部：

```python
if torch.is_grad_enabled():
    raise ValueError("Expected grad to be disabled.")
```

微调第一步 forward 就崩。修法（`datasets_tools/patch_sam3_train.py`，幂等）：

```python
if torch.is_grad_enabled():
    x = self.act(self.fc1(x))      # 训练：等价的手写路径
else:
    x = addmm_act(type(self.act), self.fc1, x)   # 推理：保留融合，不影响性能
```

### 坑 4：`semantic_seg_head` 未参与损失 → DDP 第二迭代报错

```
RuntimeError: Expected to have finished reduction in the prior iteration...
Parameter indices which did not receive grad for rank 0: 1088 1089
```

索引 1088/1089 = `segmentation_head.semantic_seg_head.{weight,bias}`：它只在启用
`loss_fn_semantic_seg` 时有梯度，而我们只用实例 mask loss。修法：`_create_sam3_model`
里 `seg_sem.requires_grad_(False)`（同在 patch 脚本内）。

### 坑 5：val 流程缺 `DecodeRle` → `KeyError: (None, None)`

官方 roboflow/odinw 的 `val_transforms` 里没有 `DecodeRle`（它们验证时 `enable_segmentation: False`），
但我们验证要带 mask，`RandomResizeAPI` 会去 resize `obj.segment`：

```
obj.segment = F.resize(obj.segment[None, None], size).squeeze()
KeyError: (None, None)
```

修法：`val_transforms` 最前面补 `sam3.train.transforms.segmentation.DecodeRle`。

### 坑 6：后处理器必须显式 `iou_type: "segm"`

`PostProcessImage` 只有在 `iou_type == "segm"` 时才产出 `prediction["masks"]`，
而 `PredictionDumper.prepare_for_coco_segmentation` 直接取 `prediction["masks"]`：

```
KeyError: 'masks'
```

修法：`scratch.original_box_postprocessor` 里加 `iou_type: "segm"`，
同时 `use_original_sizes_mask: true`（否则 mask 尺寸不对，AP 无意义）。

### 附带：云端原来只有 val 图像

`data_plantseg34/images/` 下只有 `val`（零样本评测时传的）。S2 需要 train：
5367 张 jpg 共 688 MB，tar 后一次 scp 上传（约 18 MB/s，38 s）。

## 5. 已验证的链路（冒烟）

16 张图 / 1 epoch / 8 张验证：

- 训练 16 步走完，mask/dice loss 均正常回传
- 验证走完 COCO `segm` 评测，产出 AP/AR 全项
- `checkpoints/checkpoint.pt` 正常保存（9.4 GB，含优化器状态）

## 6. S2 正式实验设置

| 项 | 值 |
| --- | --- |
| 训练集 | `plantseg34_train.json`，5367 图 / 38496 实例 |
| 提示词 | 寄主级 `disease lesions on {host} leaves`（与 S1 的 `host_0` 一致，便于隔离"微调"单一变量） |
| epoch | 10（`val_epoch_freq: 2`） |
| 学习率 | `lr_scale 0.1` → transformer 8e-5 / vision 2.5e-5 / language 5e-6 |
| 损失 | bbox + giou + IA-BCE(+presence) + mask(focal 200/dice 10) |
| 分辨率 | 1008（与 S1 零样本一致） |
| 日志 | `/root/autodl-tmp/Ryan/runs/sam3_s2_full_ft_host/` |

预计 10 epoch ≈ 22 h。

## 7. 训练完怎么评（与 S1 同口径）

训练产物 `checkpoints/checkpoint.pt` 的键**不带前缀**（`backbone.…`），
而发布权重 `sam3.pt` 是扁平 dict、键带 `detector.` / `tracker.` 前缀，推理侧按前缀解析。
所以必须先转换：

```bash
python datasets_tools/convert_ft_ckpt.py \
    --ft    /root/autodl-tmp/Ryan/runs/sam3_s2_full_ft_host/checkpoints/checkpoint.pt \
    --base  /root/autodl-tmp/Ryan/sam3.pt \
    --out   /root/autodl-tmp/Ryan/runs/sam3_s2_full_ft_host/sam3_s2.pt
```

- 训练模型没有 `backbone.vision_backbone.sam2_convs.*`（那是实例交互分支，
  `enable_inst_interactivity=False` 时不构建），转换时这 22 个键沿用 base 原值，
  推理侧才能按原结构加载。
- 已用冒烟权重实测：转换后 Ultralytics `SAM3SemanticPredictor` 能正常加载并出 mask。

评测命令（与 S1 完全相同的脚本、imgsz、conf、提示词）：

```bash
python -m sam3_baseline.prompt_sweep \
    --data-root /root/autodl-tmp/Ryan/data_plantseg34 \
    --ckpt /root/autodl-tmp/Ryan/runs/sam3_s2_full_ft_host/sam3_s2.pt \
    --variants host_0 --imgsz 1008 --conf 0.01 \
    --out experiments/sam3_s2
```


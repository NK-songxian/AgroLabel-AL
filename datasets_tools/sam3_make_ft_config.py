"""生成 SAM3 官方训练管线（sam3/train/train.py）所需的 Hydra 配置。

把 PlantSeg-34 已生成的 COCO（绝对像素坐标 + noun_phrase）直接接入官方
`Sam3ImageDataset`，无需重排目录结构 —— 官方 config 里 img_folder / ann_file
本来就是两个独立字段。

产出 yaml 必须放在 `sam3/train/configs/` 下（train.py 用
`initialize_config_module("sam3.train")`，config_name 相对该模块目录解析）。

用法（本地生成后 scp 到云端）：
    python datasets_tools/sam3_make_ft_config.py \
        --exp-name s2_full_ft_host --out configs/sam3/plantseg34_s2.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 全部改为仓库相对路径：本地/云端同一份代码都能跑（数据/权重/第三方源码放在仓库内）
DEF_ROOT = str(REPO_ROOT / "data" / "plantseg34")
DEF_COCO = str(REPO_ROOT / "datasets" / "sam3")
# 默认把 run 写进项目内 runtime/runs
DEF_LOGDIR = str(REPO_ROOT / "runtime" / "runs")
DEF_CKPT = str(REPO_ROOT / "weights" / "sam3.pt")
DEF_BPE = str(REPO_ROOT / "third_party" / "sam3_repo" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")


def build_prompts(bank_path: Path, phrase_index: int, level: str) -> str:
    """把 host_prompt_bank 里每个寄主选定的一条短语拼成 COCO_FROM_JSON 要的 JSON 字符串。

    host_prompt_bank.hosts[k]["class_id"] 是 **0 基**（0..33），COCO category id 是
    **1 基**（1..34）；这里按 1 基输出，与 plantseg34_*.json 的 categories 对齐。
    """
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    prompts = []
    for host in bank["hosts"]:
        cid = int(host["class_id"]) + 1
        if level == "host":
            phrases = host["host_level_phrases"]
        else:  # disease：取该寄主第一条病害的第一条短语（仅供单病种冒烟）
            phrases = host["diseases"][0]["phrases"]
        prompts.append({"id": cid, "name": phrases[phrase_index], "supercategory": host["host"]})
    prompts.sort(key=lambda d: d["id"])
    return json.dumps(prompts, ensure_ascii=False)


TEMPLATE = """# @package _global_
# 由 datasets_tools/sam3_make_ft_config.py 生成，不要手改。
# 实验名: {exp_name}
# 提示词级别: {level} / phrase_index={phrase_index}
# 说明: 数据侧为 PlantSeg-34（34 寄主，实例目标=病斑 region），已转 COCO 绝对像素坐标。
defaults:
  - _self_

paths:
  plantseg_root: {root}
  coco_root: {coco}
  experiment_log_dir: {logdir}
  bpe_path: {bpe}
  pretrained_ckpt: {ckpt}

plantseg_train:
  supercategory: plantseg34
  num_images: {limit_ids}
  prompts: '{prompts}'

  train_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterCrowds
        - _target_: sam3.train.transforms.point_sampling.RandomizeInputBbox
          box_noise_std: 0.1
          box_noise_max: 20
        - _target_: sam3.train.transforms.segmentation.DecodeRle
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes:
            _target_: sam3.train.transforms.basic.get_random_resize_scales
            size: ${{scratch.resolution}}
            min_size: 480
            rounded: false
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${{scratch.resolution}}
          square: true
          consistent_transform: ${{scratch.consistent_transform}}
        - _target_: sam3.train.transforms.basic_for_api.PadToSizeAPI
          size: ${{scratch.resolution}}
          consistent_transform: ${{scratch.consistent_transform}}
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${{scratch.train_norm_mean}}
          std: ${{scratch.train_norm_std}}
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
    - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
      query_filter:
        _target_: sam3.train.transforms.filter_query_transforms.FilterFindQueriesWithTooManyOut
        max_num_objects: ${{scratch.max_ann_per_img}}

  val_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        # 官方 roboflow/odinw 的 val 流程没有 DecodeRle（它们验证时不带 mask），
        # 我们做分割评测必须先把 RLE 解成张量，否则 resize 会 KeyError: (None, None)。
        - _target_: sam3.train.transforms.segmentation.DecodeRle
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes: ${{scratch.resolution}}
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${{scratch.resolution}}
          square: true
          consistent_transform: False
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${{scratch.train_norm_mean}}
          std: ${{scratch.train_norm_std}}

  loss:
    _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
    matcher: ${{scratch.matcher}}
    o2m_weight: 2.0
    o2m_matcher:
      _target_: sam3.train.matcher.BinaryOneToManyMatcher
      alpha: 0.3
      threshold: 0.4
      topk: 4
    use_o2m_matcher_on_o2m_aux: false
    loss_fns_find:
      - _target_: sam3.train.loss.loss_fns.Boxes
        weight_dict:
          loss_bbox: 5.0
          loss_giou: 2.0
      - _target_: sam3.train.loss.loss_fns.IABCEMdetr
        weak_loss: False
        weight_dict:
          loss_ce: 20.0
          presence_loss: 20.0
        pos_weight: 10.0
        alpha: 0.25
        gamma: 2
        use_presence: True
        pos_focal: false
        pad_n_queries: 200
        pad_scale_pos: 1.0
      - _target_: sam3.train.loss.loss_fns.Masks
        focal_alpha: 0.25
        focal_gamma: 2.0
        weight_dict:
          loss_mask: 200.0
          loss_dice: 10.0
        compute_aux: false
    loss_fn_semantic_seg: null
    scale_by_find_batch_size: ${{scratch.scale_by_find_batch_size}}

scratch:
  enable_segmentation: True
  d_model: 256
  pos_embed:
    _target_: sam3.model.position_encoding.PositionEmbeddingSine
    num_pos_feats: ${{scratch.d_model}}
    normalize: true
    scale: null
    temperature: 10000

  use_presence_eval: True
  original_box_postprocessor:
    _target_: sam3.eval.postprocessors.PostProcessImage
    # 必须显式给 segm：PostProcessImage 仅在 iou_type=="segm" 时才产出 masks 字段，
    # 否则 PredictionDumper.prepare_for_coco_segmentation 会 KeyError: 'masks'。
    iou_type: "segm"
    max_dets_per_img: -1
    use_original_ids: true
    use_original_sizes_box: true
    use_original_sizes_mask: true
    use_presence: ${{scratch.use_presence_eval}}

  matcher:
    _target_: sam3.train.matcher.BinaryHungarianMatcherV2
    focal: true
    cost_class: 2.0
    cost_bbox: 5.0
    cost_giou: 2.0
    alpha: 0.25
    gamma: 2
    stable: False
  scale_by_find_batch_size: True

  resolution: {resolution}
  consistent_transform: False
  max_ann_per_img: 200

  train_norm_mean: [0.5, 0.5, 0.5]
  train_norm_std: [0.5, 0.5, 0.5]
  val_norm_mean: [0.5, 0.5, 0.5]
  val_norm_std: [0.5, 0.5, 0.5]

  num_train_workers: {train_workers}
  num_val_workers: 2
  max_data_epochs: {max_epochs}
  target_epoch_size: 1500
  hybrid_repeats: 1
  gather_pred_via_filesys: false

  # 微调尺度：官方 FT 配方用 lr_scale=0.1
  lr_scale: {lr_scale}
  lr_transformer: ${{times:8e-4,${{scratch.lr_scale}}}}
  lr_vision_backbone: ${{times:2.5e-4,${{scratch.lr_scale}}}}
  lr_language_backbone: ${{times:5e-5,${{scratch.lr_scale}}}}
  lrd_vision_backbone: 0.9
  wd: 0.1
  scheduler_timescale: 20
  scheduler_warmup: 20
  scheduler_cooldown: 20

  val_batch_size: 1
  collate_fn_val:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${{scratch.hybrid_repeats}}
    dict_key: plantseg34
    with_seg_masks: ${{scratch.enable_segmentation}}

  gradient_accumulation_steps: {grad_accum}
  train_batch_size: 1
  collate_fn:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${{scratch.hybrid_repeats}}
    dict_key: all
    with_seg_masks: ${{scratch.enable_segmentation}}
  # 每个 datapoint 携带的类别查询数。1=每图只问 1 个类别（最省显存，图会重复 34 次）；
  # 34=每图一次问满 34 类（含负样本，最贵）。
  category_chunk_size: {chunk}

trainer:
  _target_: sam3.train.trainer.Trainer
  skip_saving_ckpts: false
  empty_gpu_mem_cache_after_eval: true
  skip_first_val: true
  max_epochs: {max_epochs}
  accelerator: cuda
  seed_value: 123
  val_epoch_freq: {val_freq}
  mode: train
  gradient_accumulation_steps: ${{scratch.gradient_accumulation_steps}}

  distributed:
    backend: nccl
    find_unused_parameters: True
    gradient_as_bucket_view: True

  loss:
    all: ${{plantseg_train.loss}}
    default:
      _target_: sam3.train.loss.sam3_loss.DummyLoss

  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        limit_ids: ${{plantseg_train.num_images}}
        transforms: ${{plantseg_train.train_transforms}}
        load_segmentation: ${{scratch.enable_segmentation}}
        max_ann_per_img: 500000
        multiplier: 1
        max_train_queries: 50000
        max_val_queries: 50000
        training: true
        use_caching: False
        img_folder: ${{paths.plantseg_root}}/images/train/
        ann_file: ${{paths.coco_root}}/plantseg34_train.json
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          prompts: ${{plantseg_train.prompts}}
          include_negatives: {train_neg}
          category_chunk_size: ${{scratch.category_chunk_size}}
          _partial_: true
      shuffle: True
      batch_size: ${{scratch.train_batch_size}}
      num_workers: ${{scratch.num_train_workers}}
      pin_memory: True
      drop_last: True
      collate_fn: ${{scratch.collate_fn}}

    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        limit_ids: {val_images}
        transforms: ${{plantseg_train.val_transforms}}
        load_segmentation: ${{scratch.enable_segmentation}}
        max_ann_per_img: 100000
        multiplier: 1
        max_train_queries: 50000
        max_val_queries: 50000
        training: false
        use_caching: False
        img_folder: ${{paths.plantseg_root}}/images/val/
        ann_file: ${{paths.coco_root}}/plantseg34_val.json
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          prompts: ${{plantseg_train.prompts}}
          include_negatives: false
          category_chunk_size: ${{scratch.category_chunk_size}}
          _partial_: true
      shuffle: False
      batch_size: ${{scratch.val_batch_size}}
      num_workers: ${{scratch.num_val_workers}}
      pin_memory: True
      drop_last: False
      collate_fn: ${{scratch.collate_fn_val}}

  model:
    _target_: {model_target}
    bpe_path: ${{paths.bpe_path}}
    checkpoint_path: ${{paths.pretrained_ckpt}}
    load_from_HF: false
    device: cpus
    eval_mode: false
    enable_segmentation: ${{scratch.enable_segmentation}}{model_extra}

  meters:
    val:
      plantseg34:
        segm:
          _target_: sam3.eval.coco_writer.PredictionDumper
          iou_type: "segm"
          dump_dir: ${{launcher.experiment_log_dir}}/dumps/plantseg34/segm
          merge_predictions: True
          postprocessor: ${{scratch.original_box_postprocessor}}
          gather_pred_via_filesys: ${{scratch.gather_pred_via_filesys}}
          maxdets: 100
          pred_file_evaluators:
            - _target_: sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators
              gt_path: ${{paths.coco_root}}/plantseg34_val.json
              tide: False
              iou_type: "segm"

  optim:
    amp:
      enabled: True
      amp_dtype: bfloat16

    optimizer:
      _target_: torch.optim.AdamW

    gradient_clip:
      _target_: sam3.train.optim.optimizer.GradientClipper
      max_norm: 0.1
      norm_type: 2

    param_group_modifiers:
      - _target_: sam3.train.optim.optimizer.layer_decay_param_modifier
        _partial_: true
        layer_decay_value: ${{scratch.lrd_vision_backbone}}
        apply_to: 'backbone.vision_backbone.trunk'
        overrides:
          - pattern: '*pos_embed*'
            value: 1.0

    options:
      lr:
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_transformer}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_vision_backbone}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
          param_names:
            - 'backbone.vision_backbone.*'
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_language_backbone}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
          param_names:
            - 'backbone.language_backbone.*'

      weight_decay:
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: ${{scratch.wd}}
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: 0.0
          param_names:
            - '*bias*'
          module_cls_names: ['torch.nn.LayerNorm']

  checkpoint:
    save_dir: ${{launcher.experiment_log_dir}}/checkpoints
    save_freq: 0

  logging:
    tensorboard_writer:
      _target_: sam3.train.utils.logger.make_tensorboard_logger
      log_dir: ${{launcher.experiment_log_dir}}/tensorboard
      flush_secs: 120
      should_log: True
    wandb_writer: null
    log_dir: ${{launcher.experiment_log_dir}}/logs/${{plantseg_train.supercategory}}
    log_freq: 10

launcher:
  num_nodes: 1
  gpus_per_node: 1
  experiment_log_dir: ${{paths.experiment_log_dir}}
  multiprocessing_context: forkserver

submitit:
  account: null
  partition: null
  qos: null
  timeout_hour: 72
  use_cluster: False
  cpus_per_task: 10
  port_range: [10000, 65000]
  constraint: null
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", default="s2_full_ft_host")
    ap.add_argument("--out", default="configs/sam3/plantseg34_s2.yaml")
    ap.add_argument("--host-bank", default="configs/sam3/host_prompt_bank.json")
    ap.add_argument("--level", default="host", choices=["host", "disease"])
    ap.add_argument("--phrase-index", type=int, default=0)
    ap.add_argument("--root", default=DEF_ROOT)
    ap.add_argument("--coco", default=DEF_COCO)
    ap.add_argument("--logdir", default=DEF_LOGDIR)
    ap.add_argument("--ckpt", default=DEF_CKPT)
    ap.add_argument("--bpe", default=DEF_BPE)
    ap.add_argument("--limit-ids", type=int, default=None, help="训练图数上限，冒烟用")
    ap.add_argument("--val-images", type=int, default=100, help="训练中验证用图数上限")
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--val-freq", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=1008)
    # trainer._run_step 在 grad_accum>1 时要求 dataloader 直接吐 list[N]，
    # 而 TorchDataset/DataLoader 只会吐单个 batch dict —— 官方 recipe 也是 1，故固定 1。
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr-scale", type=float, default=0.1)
    ap.add_argument("--train-workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=34, help="每图类别查询数")
    ap.add_argument("--train-neg", default="true", choices=["true", "false"])
    # ---- LoRA（I-01/I-02）：rank=0 表示不注入，走官方全参微调 ----
    ap.add_argument("--model-target", default="",
                    help="model._target_；留空则按 lora-rank 自动选官方 builder 或 src.lora_builder.build_sam3_lora")
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-targets", default="",
                    help="逗号分隔的模块名子串（留空=全部 Linear 都注入）")
    ap.add_argument("--lora-extra-trainable", default="",
                    help="除 lora_* 外额外放开训练的模块名子串，如 mask_decoder")
    args = ap.parse_args()

    bank = REPO_ROOT / args.host_bank
    if not bank.exists():
        raise SystemExit(f"找不到提示词库: {bank}")
    prompts = build_prompts(bank, args.phrase_index, args.level)

    logdir = os.path.join(args.logdir, f"sam3_{args.exp_name}")

    OFFICIAL = "sam3.model_builder.build_sam3_image_model"
    LORA = "src.lora_builder.build_sam3_lora"
    model_target = args.model_target or (LORA if args.lora_rank > 0 else OFFICIAL)
    model_extra = ""
    if args.lora_rank > 0 or "lora_builder" in model_target:
        model_extra = (
            f"\n    lora_rank: {args.lora_rank}"
            f"\n    lora_alpha: {args.lora_alpha}"
            f"\n    lora_dropout: {args.lora_dropout}"
            f"\n    lora_targets: '{args.lora_targets}'"
            f"\n    lora_extra_trainable: '{args.lora_extra_trainable}'"
        )

    text = TEMPLATE.format(
        exp_name=args.exp_name,
        level=args.level,
        phrase_index=args.phrase_index,
        root=args.root,
        coco=args.coco,
        logdir=logdir,
        bpe=args.bpe,
        ckpt=args.ckpt,
        limit_ids=args.limit_ids if args.limit_ids else "null",
        prompts=prompts,
        resolution=args.resolution,
        grad_accum=args.grad_accum,
        lr_scale=args.lr_scale,
        train_workers=args.train_workers,
        max_epochs=args.max_epochs,
        val_freq=args.val_freq,
        chunk=args.chunk,
        train_neg=args.train_neg,
        val_images=args.val_images,
        model_target=model_target,
        model_extra=model_extra,
    )

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[OK] 写入 {out}")
    print(f"     实验名 {args.exp_name} -> 日志 {logdir}")
    print(f"     model._target_ = {model_target}")
    if args.lora_rank > 0:
        print(f"     LoRA rank={args.lora_rank} alpha={args.lora_alpha} "
              f"targets='{args.lora_targets or 'ALL'}' extra='{args.lora_extra_trainable}'")
    print(f"     提示词级别 {args.level} phrase#{args.phrase_index}，共 "
          f"{len(json.loads(prompts))} 类")


if __name__ == "__main__":
    main()

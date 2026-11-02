"""Task-SAM3-001 · 二、构建 host_prompt_bank.json

规则：
- 34 类是寄主（host），**不得把 host 名称直接当病害名**。
- 每个寄主给 3 条标准病斑级候选短语（任务指定），外加 1 条中性命名词（避免器官假设）。
- 若元数据含具体病害名（115 类 JSON 的 Disease 字段，且被文件名 100% 覆盖），
  额外建立病害级短语。
- 每张图只取自身寄主（及自身病害）的短语，不做 34 类遍历。
- 置信度 != 准确率：短语筛选必须用 val 真实指标，不能用 conf 排序。

用法：
  python datasets_tools/sam3_build_prompt_bank.py
"""
import argparse
import json
import os
from collections import Counter


def load_classes(p):
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else d["classes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="F:/_plantseg/dataset_plantseg_34")
    ap.add_argument("--meta-root", default="F:/_plantseg/dataset_plantseg_115")
    ap.add_argument("--audit", default="experiments/sam3_data_audit.json")
    ap.add_argument("--out", default="configs/sam3/host_prompt_bank.json")
    a = ap.parse_args()

    hosts = load_classes(os.path.join(a.data_root, "classes.json"))
    combos = load_classes(os.path.join(a.meta_root, "classes.json"))

    # 观察到的病害→实例数（来自审计，用于排序，不用于臆造）
    obs = Counter()
    if os.path.exists(a.audit):
        with open(a.audit, encoding="utf-8") as f:
            audit = json.load(f)
        for c in audit["classes"]:
            for d in c["diseases"]:
                obs[d["name"]] += d["instances"]
    else:
        audit = None

    disease_by_host = {}
    for c in combos:
        plant, dis = c.split("___")
        disease_by_host.setdefault(plant, []).append(dis)

    bank = {
        "task": "Task-SAM3-001 · 提示词库",
        "version": "v1",
        "semantics": {
            "class_level": "host_plant",
            "instance_target": "disease_lesion_region",
            "verified_by": "experiments/sam3_data_audit.md",
        },
        "rules": {
            "no_cross_class_traversal": "每张图只用其自身 host（及自身 disease）的短语，不跑 34 类全量",
            "host_is_not_disease": "host 名称（如 apple / tomato）只能作为器官所属主体出现，"
                                   "不能单独作为病害短语；单用 host 名属于目标错配",
            "confidence_is_not_accuracy": "模型置信度不能代替真实准确率，候选短语的取舍"
                                          "必须以 val 上的 mask AP / Dice 为准",
            "organ_not_assumed": "短语中的 leaves 为任务给定形式；审计显示部分实例位于果实，"
                                 "器官词需按 val 实测结果调整，不得默认",
            "no_fabrication": "所有病害名均来自 dataset_plantseg_115/classes.json 的 Disease 字段，"
                              "且被文件名前缀 100% 覆盖，无人工补充",
        },
        "baselines": {
            "S0_zero_shot_species_prompt": {
                "file": "experiments/sam3_probe_p0.md",
                "setting": "裸类名（apple / soybean …）零样本，conf=0.01，imgsz=1008",
                "hit_rate": "52.2%（70/134）",
                "note": "提示词指向整株/整果，与病斑标注目标错配，仅作基线保留",
            },
            "S1_zero_shot_tuned_species": {
                "file": "experiments/sam3_probe_p0.md",
                "setting": "每类 best-of-5 变体（含 leaf/plant/leaves），val 846 张全量",
                "mask_mAP50": 0.0254,
                "dice_matched": 0.8449,
                "note": "仍为物种级提示，未针对病斑；保留为 S1",
            },
        },
        "prompt_templates": {
            "host_level": [
                "disease lesions on {host} leaves",
                "leaf disease spots on {host} leaves",
                "abnormal diseased regions on {host} leaves",
                "disease lesions on {host}",
            ],
            "disease_level": [
                "{disease} lesions on {host} leaves",
                "{disease} lesions on {host}",
                "{disease} spots on {host} fruit",
            ],
        },
        "hosts": [],
    }

    for i, h in enumerate(hosts):
        hl_phrases = [t.format(host=h.lower()) for t in bank["prompt_templates"]["host_level"]]
        dis_list = sorted(disease_by_host.get(h, []), key=lambda d: -obs.get(d, 0))
        entry = {
            "class_id": i, "host": h,
            "host_level_phrases": hl_phrases,
            "diseases": [{
                "name": d,
                "observed_instances": int(obs.get(d, 0)),
                "phrases": [t.format(host=h.lower(), disease=d.lower())
                            for t in bank["prompt_templates"]["disease_level"]],
            } for d in dis_list],
        }
        bank["hosts"].append(entry)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    n_d = sum(len(e["diseases"]) for e in bank["hosts"])
    n_p = sum(len(e["host_level_phrases"]) + sum(len(d["phrases"]) for d in e["diseases"])
              for e in bank["hosts"])
    print(f"[bank] {len(bank['hosts'])} 寄主 / {n_d} 病害 / {n_p} 条候选短语 -> {a.out}")
    e0 = bank["hosts"][0]
    print(f"[bank] 示例 {e0['host']}: host={e0['host_level_phrases'][0]!r} "
          f"disease={e0['diseases'][0]['phrases'][0]!r}")


if __name__ == "__main__":
    main()

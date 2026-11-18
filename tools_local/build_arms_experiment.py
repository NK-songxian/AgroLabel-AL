# -*- coding: utf-8 -*-
"""三臂对照实验数据包构建（v1.1，2026-11-17）：
臂A manual   —— 纯人工，空白画布
臂B assisted —— B6 预标注 + 人工编辑（实时调 /api/predict）
臂C noprompt —— B6 预标注 + 人工编辑，但禁用点提示（交互式精修消融）

注意：AL 不确定性排序不属于三臂对照，它只进 m8_al.py 训练侧 4 策略消融；
前端「智能选样」只用于演示截图/录屏，不进本实验统计。

产出（F:/AgroLabel-AL/deliverables/exp60/）：
- manifest.json           图清单（val 分层抽样）+ 每图唯一臂分配
- predictions_b6.json     B6 预标注备份（队友无需 GPU；实验主流程用实时 API）
- gt_labels_val/          对应 GT（YOLO txt，评分对照必需）
- timing_sheet.csv        计时备份表（主口径用后端 ops 日志自动记录）
需本地后端已启动（B6 权重）。路径可用环境变量覆盖，便于队友机器复现。
"""
import json
import os
import random
import shutil
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import yaml

DATA = Path(os.environ.get('AGROLABEL_DATA_ROOT', 'F:/_plantseg/dataset_plantseg_34'))
OUT = Path(os.environ.get('AGROLABEL_EXP60_OUT', 'F:/AgroLabel-AL/deliverables/exp60'))
OUT.mkdir(parents=True, exist_ok=True)
API = os.environ.get('AGROLABEL_API', 'http://127.0.0.1:8000')
ARMS = ('manual', 'assisted', 'noprompt')  # 与后端 VALID_ARMS 严格一致

# ---- 类别表 ----
dy = yaml.safe_load((DATA / 'data.yaml').read_text(encoding='utf-8'))
names = dy['names']
if isinstance(names, dict):
    names = [names[i] for i in range(len(names))]

# ---- catalog：取 val split 的图（有 GT、规模适中），分页拉全（单页上限 200）----
items, offset = [], 0
while True:
    cat = json.load(urllib.request.urlopen(
        f'{API}/api/images?split=val&limit=200&offset={offset}', timeout=60))
    items.extend(cat['items'])
    offset += len(cat['items'])
    if offset >= cat['total'] or not cat['items']:
        break
assert cat['total'] == len(items), f"分页未取全: {len(items)}/{cat['total']}"
print(f'catalog val 返回 {len(items)} 图')

# ---- 按 GT 宿主类分层抽样：每类随机 2 张（可复现，避免 catalog 字母序偏差）----
by_cls = defaultdict(list)
for it in items:
    cid = it.get('gt_class_id')
    if cid is None:
        continue
    by_cls[cid].append(it)
picked = []
for cid in sorted(by_cls):
    pool = by_cls[cid]
    picked.extend(random.Random(42 + int(cid)).sample(pool, min(2, len(pool))))
print(f'分层抽样 {len(picked)} 图（{len(by_cls)} 类）')

# ---- 每图只进一个臂：按类内交错轮转，保证同一类的两张图不落在同一臂 ----
assign = []
for k, cid in enumerate(sorted(by_cls)):
    class_imgs = [it for it in picked if it.get('gt_class_id') == cid]
    for j, it in enumerate(class_imgs):
        arm = ARMS[(k + j) % len(ARMS)]
        it['_arm'] = arm
        assign.append(it)
picked = sorted(assign, key=lambda x: x['image_id'])
from collections import Counter
print('臂分配：', dict(Counter(it['_arm'] for it in picked)))

# ---- 批量预标注（本地 B6 后端）----
preds, lat = {}, []
fail = []
t0 = time.time()
for it in picked:
    iid = it['image_id']
    req = urllib.request.Request(f'{API}/api/predict',
        data=json.dumps({'image_id': iid}).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=120))
        preds[iid] = d['instances']
        lat.append(d['latency_ms'])
    except Exception as e:
        fail.append(iid)
print(f'预标注完成 {len(preds)}/{len(picked)}，平均 {sum(lat)/max(1,len(lat)):.0f}ms，'
      f'失败 {len(fail)}: {fail[:5]}')

# ---- manifest ----
manifest = {
    'experiment': '三臂标注效率对照（纯人工 / B6预标注+编辑 / B6预标注+编辑+禁用点提示）',
    'date': '2026-11-17',
    'model': 'b6_y26x_1280_best.pt (val mask mAP50 0.416)',
    'sampling': 'val 846 图按 GT 宿主类分层，每类随机取 2 张（seed=42+class_id）；按类交错轮转分臂',
    'n_images': len(picked),
    'classes': names,
    'arms': {
        'A': 'manual：纯人工，空白画布从零画掩码+选类',
        'B': 'assisted：B6 预标注为底稿，人工拖边/删除/补漏/改类；点提示可用',
        'C': 'noprompt：同 B，但禁用点提示（交互式精修消融）',
    },
    'backend_arms': {'A': 'manual', 'B': 'assisted', 'C': 'noprompt'},
    'assignment_rule': '每张图只进一个臂；同类的 2 张图分到不同臂；主计时用后端 ops 日志，timing_sheet.csv 仅备份',
    'timing_rule': '以"开始查看该图"到"点击提交"的秒数计；后端 elapsed_from_open_ms 为主口径，手工表仅备份',
    'images': [{'image_id': it['image_id'], 'file_name': it['file_name'],
                'gt_class_id': it['gt_class_id'], 'gt_class_name': it['gt_class_name'],
                'arm_backend': it['_arm'],
                'width': it['width'], 'height': it['height']} for it in picked],
    'pred_stats': {'n_fail': len(fail),
                   'avg_latency_ms': round(sum(lat) / max(1, len(lat)), 1)},
}
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                   encoding='utf-8')
(OUT / 'predictions_b6.json').write_text(json.dumps(preds, ensure_ascii=False),
                                         encoding='utf-8')

# ---- GT labels 拷贝 ----
gtdir = OUT / 'gt_labels_val'
gtdir.mkdir(exist_ok=True)
n_gt = 0
for it in picked:
    src = DATA / 'labels' / 'val' / (Path(it['file_name']).stem + '.txt')
    if src.exists():
        shutil.copy2(src, gtdir / src.name)
        n_gt += 1
print(f'GT labels 拷贝 {n_gt}/{len(picked)}')

# ---- 计时表模板（备份/交叉核对；主口径是后端 ops 自动日志）----
rows = ['record_id,image_id,arm_backend,annotator,start_time,end_time,duration_s,'
        'ops_add,ops_delete,ops_drag,ops_reclass,notes']
for i, it in enumerate(picked):
    rows.append(f'{i+1:03d},{it["image_id"]},{it["_arm"]},,,, ,,,,,')
(OUT / 'timing_sheet.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
print('输出完成 ->', OUT)

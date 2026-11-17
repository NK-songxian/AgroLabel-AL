# -*- coding: utf-8 -*-
"""三臂对照实验数据包构建：
臂A 纯人工 / 臂B B6预标注+编辑 / 臂C B6预标注+编辑+AL排序。

产出（F:/AgroLabel-AL/deliverables/exp60/）：
- manifest.json           图清单（分层抽样，每类>=2 张 val 图）+ 臂分配
- predictions_b6.json     全部实验图的 B6 预标注（臂B/C 的起点，队友无需 GPU）
- gt_labels/              对应 GT（YOLO txt，评分对照必需）
- timing_sheet.csv        计时记录表模板
需本地后端已启动（B6 权重）。
"""
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import yaml

DATA = Path('F:/_plantseg/dataset_plantseg_34')
OUT = Path('F:/AgroLabel-AL/deliverables/exp60')
OUT.mkdir(parents=True, exist_ok=True)
API = 'http://127.0.0.1:8000'

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

# ---- 按 GT 宿主类分层抽样：每类 2 张 ----
by_cls = defaultdict(list)
for it in items:
    by_cls[it['gt_class_id']].append(it)
picked = []
for cid in sorted(by_cls):
    picked.extend(by_cls[cid][:2])
print(f'分层抽样 {len(picked)} 图（{len(by_cls)} 类）')

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
    'experiment': '三臂标注效率对照（纯人工 / B6预标注+编辑 / B6预标注+编辑+AL排序）',
    'date': '2026-11-17',
    'model': 'b6_y26x_1280_best.pt (val mask mAP50 0.416)',
    'sampling': 'val 846 图按 GT 宿主类分层，每类取前 2 张',
    'n_images': len(picked),
    'classes': names,
    'arms': {
        'A': '纯人工：从零画掩码+选类',
        'B': 'B6 预标注加载为底稿，人工修正后提交',
        'C': '同 B，但图序按模型不确定性（entropy）降序处理——先做模型最没把握的图',
    },
    'timing_rule': '以"开始查看该图"到"点击提交"的秒数计；中途离开需暂停记录',
    'images': [{'image_id': it['image_id'], 'file_name': it['file_name'],
                'gt_class_id': it['gt_class_id'], 'gt_class_name': it['gt_class_name'],
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
import shutil
n_gt = 0
for it in picked:
    src = DATA / 'labels' / 'val' / (Path(it['file_name']).stem + '.txt')
    if src.exists():
        shutil.copy2(src, gtdir / src.name)
        n_gt += 1
print(f'GT labels 拷贝 {n_gt}/{len(picked)}')

# ---- 计时表模板 ----
rows = ['record_id,image_id,arm,annotator,start_time,end_time,duration_s,'
        'ops_add,ops_delete,ops_drag,ops_reclass,notes']
for i, it in enumerate(picked):
    rows.append(f'{i+1:03d},{it["image_id"]},A或B或C,,                ,,,,,,,')
(OUT / 'timing_sheet.csv').write_text('\n'.join(rows), encoding='utf-8')
print('输出完成 ->', OUT)

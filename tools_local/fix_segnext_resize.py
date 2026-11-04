# -*- coding: utf-8 -*-
"""把 plantseg34 segnext 配置里的 RandomResize 换成固定 Resize(512,512)。"""
import re

p = '/root/autodl-tmp/Ryan/ArgoLable-AL/configs_mmseg/segnext/segnext_mscan-l_plantseg34-40k.py'
s = open(p).read()
pat = re.compile(
    r"dict\(\s*keep_ratio=True,\s*ratio_range=\(\s*0\.5,\s*2\.0,\s*\),\s*"
    r"scale=\(\s*2048,\s*512,\s*\),\s*type='RandomResize'\),")
new = "dict(scale=(512, 512), keep_ratio=False, type='Resize'),"
s2, n = pat.subn(new, s, count=1)
assert n == 1, 'no substitution'
open(p, 'w').write(s2)

from mmengine.config import Config
c = Config.fromfile(p)
pipe = c.train_dataloader.dataset['pipeline']
kinds = [t['type'] for t in pipe]
assert 'RandomResize' not in kinds, kinds
print('patched OK, train pipeline =', kinds)

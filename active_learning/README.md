# active_learning/

VARD-AL 主动学习采样策略与评估。

| 功能 | 旧脚本 |
|------|--------|
| 主动学习闭环 | `src/m8_al.py` |
| 长尾类别分析 | `src/m7_longtail.py` |

核心：联合采样价值
`S_i = α·U_i + β·D_i + γ·R_i + δ·G_i − ε·Ĉ_i`
（U 不确定性 / D 多样性 / R 稀有类别 / G 模型分歧 / Ĉ 归一化标注成本）

配置见 `configs/active_learning.yaml`；对比实验必须包含**随机采样对照**。

## 入口（已实现）

`active_learning/select.py` 提供交接文档 §6 约定的签名，`/api/sample` 直接调：

```python
from active_learning.select import select

picked = select(pool, k=5, strategy="entropy")  # strategy: random|entropy|tta|coreset
```

策略实现**复用 `src/m8_al.py` 的 `score_*`**，保证在线接口与离线消融同口径。
`strategy="coreset"` 可传 `labeled=[...]` 让首批选择偏向「和已有数据最不像」的样本。


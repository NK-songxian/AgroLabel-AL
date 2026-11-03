# 待办与已定事项

> 记在这里的原因：口头/聊天里定的东西，换个会话就没了。
> 每条都要能回答"谁定的、什么时候、怎么算完成"。

---

## 1. ⏳ 34 类 → 115 类（等前后端验收通过后做）

**状态**：暂缓，不是现在做。

**背景**：`baseline_yolov8nseg_plantseg34_best.pt` 是**最小可行性实验**用的权重，
不是最终形态。团队已把 115 类（`物种___病害` 组合粒度）的映射表收进仓库：
`datasets/plantseg115/classes.json`（`mode: "combo"`）。

**决定**（2026-11-02，团队）：

> 先用 34 类把前后端全部搞定、测试通过、审查之后，再改成 115 类。

**为什么不在现在改**：34 → 115 是任务定义层面的变更，会同时波及
权重、`/api/classes`、前端下拉、`out/demo` 清单、m5/m6 的全部统计口径。
MVP 阶段先冻结在 34 类，避免边做边推翻。

**切换时要一起动的**（别只改 `/api/classes`）：

| 位置 | 现状 | 切换后 |
|---|---|---|
| `backend/config.py` | `_load_classes()` 读 `configs/dataset_plantseg34.local.yaml` 的 `names` | 读 `datasets/plantseg115/classes.json` 的 `classes` |
| `backend/config.py` | `NC = 34` | `NC = 115` |
| `GET /api/classes` | 34 条 | 115 条（前端下拉自动跟着变，但会很长，要考虑分组） |
| `weights/` | `baseline_yolov8nseg_plantseg34_best.pt` | **需重训**，34 类权重识别不了 115 类 |
| `out/demo/_manifest.json` | 20 张，按 34 类映射 | 要重新生成（`build_demo_set.py` 依赖 34 类的命名） |
| `src/m5_logstats.py` | 算子统计与类别无关 | 不受影响 |
| `src/m6_editcost.py` | `breakdown` 与类别无关 | 不受影响 |
| 前端类别下拉 | 34 项可搜索 | 115 项，**建议改成"作物分组 + 搜索"**，平铺会难用 |

**完成判据**：115 类权重就位 + `/api/classes` 返回 115 条 + 前端下拉能用 + 冒烟测试全过。

**提醒我**：前后端验收通过时，主动提这件事。

---

## 2. 其他已知未完成项

（按优先级，详细说明见 `backend/README.md` 与 `frontend/README.md`）

- ~~**手画多边形 + 拖顶点**~~ ✅ 已完成（2026-11-02）：`新画` / `拖边界` 现在会真的产生
- **实验臂切换**（assisted / manual / noprompt）—— 三臂对照实验的前置
- **`GET /api/annotations/{id}`** —— 续标要用，现在关掉浏览器就丢
- **`/api/predict` 与模型侧 `inference/predict.py` 两套实现并存** —— 技术债，
  模型侧加新模块时后端不会自动跟着变
- **`/api/sample` 的 `reasons.score` 恒为 null** —— 需要模型侧 `m8_al` 的 `score_*` 返回分值
- **真实数据集 `F:/_plantseg` 未就位** —— 三臂实验与"修改次数下降 30%"的实数都出不来
- **`datasets/sam3/plantseg34_{train,val}.json` 共 24 MB 已入库** —— 占 `.git` 的 92%，
  若可由 `datasets_tools/` 再生，建议 gitignore

---

## AL-TTA-LATENCY · `tta` 选样在 100 张池上要 63 秒（**已实现 A+C**）

**发现于** 2026-11-03，接入 `datasets/plantseg100_pack`（100 张）后实测。

`m8_al.score_tta` 对每张图跑 4 次前向（尺度 1.0/1.25 × 不翻转/水平翻转）算掩码
两两 IoU。池子里 100 张 → 400 次前向 → **63 s**。前端点一下「智能选样 · TTA」
会转一分钟，演示时等同于卡死。

对比：`random` 35 ms、`coreset` 0.7 s、`entropy` 2.3 s。

**候选方案**

| 方案 | 做法 | 代价 |
|---|---|---|
| A. 两级瀑布 | 先按 entropy 粗筛到 ~2k 张，只对候选跑 TTA | 改算法语义，需在报告里写明"两级筛选"；再快也快不过 entropy 的 2.3 s |
| B. 限制扫描量 | 池子随机抽 N 张（如 40）再排序 | 简单，但会漏掉没抽到的难例，且随机性引入噪声 |
| C. 离线批处理 | TTA 结果落 `out/al_scores/<hash>.json` 缓存，按图集哈希失效 | 首次仍慢，但要等第一次跑完 |
| D. 降到 2 视图 | 只做尺度 1.0/1.25 不翻转 | 快一倍（~31 s），一致性信号变弱 |

**建议** A + C：两级瀑布把交互拉到秒级，缓存避免重复计算；报告里写清"两级"的
设计动机（先用便宜的熵粗筛、再用贵的 TTA 精排），反而是加分点——
这属于"难例筛选策略合理性"那 12 分要讲的东西。

'**顺带**：`entropy` 的 2.3 s 也不算快，同样可以用 C（缓存）压到毫秒级。

**2026-11-03 已实现 A + C**（`backend/routers/sample.py`）：
tta 先 entropy 粗筛到 max(4k,24) 再精排，分值落 `out/al_scores/<strategy>.json`。
池 36 张实测：tta 冷 24.9 s → 热 **22 ms**；entropy 冷 1.14 s → 热 **25 ms**。
冷却时间仍在（首次点击要等），要彻底消除需在启动时预热缓存 —— 未做。'

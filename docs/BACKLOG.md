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

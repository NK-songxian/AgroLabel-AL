# 前后端交接说明（给做 UI / 服务的队友）

> 项目：AgroLabel-AL —— 面向智慧农业的多模态数据智能预标注与主动学习原型系统
> 你的角色：把"模型能力"变成"能演示的标注工具"。模型、数据、训练都不用你碰。

---

## 0. 一句话

你负责 `backend/`（HTTP 服务）+ `frontend/`（标注界面），**严格按 `docs/INTERFACE.md` 冻结的契约实现**。先 mock 假数据把流程跑通，再接真模型。

---

## 1. 你负责哪几个文件夹

### ✅ 主战场（你写代码的地方）
- `backend/` —— FastAPI 服务，实现 4 个接口
- `frontend/` —— 标注界面（画布 / 修正 / 选样）

### 👀 只读（契约，别改）
- `docs/INTERFACE.md` —— **接口契约的唯一权威**。字段名、路径、出入参都必须一致。要改先找模型侧商量。

### 🔸 可能动一点
- `configs/` —— 加端口、路径等少量配置

### ❌ 严禁碰（会出问题）
- `datasets/`、`weights/`、`*.pt` —— 数据合规 + 权重几十 MB 不入库
- `training/`、`active_learning/`、`src/`、`datasets_tools/`、`experiments/`、`runs/` —— 全是模型侧的东西
- 不要提交 `node_modules/`、`dist/`、`.env`、`__pycache__/`（已在 `.gitignore`）

> 一句话记法：**`backend/` 和 `frontend/` 是你的，别的都是别人的。**

---

## 2. 接口契约（先冻结，再写代码）

来自 `docs/INTERFACE.md`，字段名一字不能改：

| 方法 | 路径 | 入参 | 出参 |
|---|---|---|---|
| POST | `/api/predict` | `{image_base64 或 image_id}` | `{instances:[{class_id, class_name, score, polygon, bbox}], latency_ms}` |
| POST | `/api/sample` | `{pool:[image_id...], k, strategy}`（`random\|entropy\|tta\|coreset`） | `{selected:[image_id...], reasons}` |
| POST | `/api/edit-cost` | `{pred_instances, gt_instances}` | `{edit_count, breakdown:{新画,删实例,改类别,拖边界,标为无目标}}` |
| GET | `/api/health` | — | `{status:"ok", model:"...", classes:34}` |

**数据格式**：YOLO seg txt（每行 `class_id x1 y1 ...` 归一化多边形）；实例 `{class_id, class_name, score, polygon, bbox}`。
**时延预算**：单张 `/api/predict` ≤ **300 ms**（模型推理参考 20–30 ms @640，以实测为准）。
**加载策略**：模型在进程启动时加载一次。

---

## 3. 技术栈选型（已定）

**结论：后端 FastAPI，前端 React + Konva。**

| 层 | 推荐 | 为什么不选其他 |
|---|---|---|
| 后端框架 | **FastAPI** | Flask 同步、无自动文档；Node/Go 需跨进程调 Python 模型，多一层序列化 |
| 画布库 | **Konva** | Fabric.js 的 React/Vue 封装较弱；原生 Canvas 要自己做命中检测 |
| 前端框架 | **React 18 + TS** | `react-konva` 最成熟；更熟 Vue 可用 Vue3 + `vue-konva` 平替 |
| 状态管理 | **Zustand** | 比 Redux 轻，标注状态不复杂 |
| 打包 | **Vite** | CRA 已停维护、启动慢 |

**依赖清单**

```bash
# 后端（backend/）
pip install fastapi "uvicorn[standard]" pydantic pillow opencv-python python-multipart ultralytics

# 前端（frontend/）
npm create vite@latest . -- --template react-ts
npm i react-konva konva zustand antd axios
```

> `node_modules/`、`dist/` 已在 `.gitignore` 中忽略，不要提交。

---

## 4. 分阶段任务（P0 → P3）

### P0 —— Mock 跑通
- [ ] `backend/app.py` 起 FastAPI，实现 4 个接口，**全部返回假数据**
- [ ] 假数据形状严格符合契约（可放 `backend/tests/mock_data.json`）
- **验收**：`curl` 四个接口都返回合法 JSON；`/api/health` 通

### P1 —— 前端画布
- [ ] 加载一批图，画布显示图片
- [ ] 叠加 `/api/predict` 的预标注：框 + 多边形（不同类别不同颜色）
- [ ] 标注工具：**画 / 拖边界 / 删 / 改类别 / 标为无目标**
- [ ] 操作日志埋点：记录每张图"改了几次、改的什么类型、耗时"
- **验收**：能在一张图上完成一次完整修正，日志能导出

### P2 —— 接真模型
- [ ] 后端接模型侧的推理 / 选样入口（见第 6 节签名）
- [ ] `/api/predict`、`/api/sample` 走真模型
- [ ] 模型在进程启动时加载一次，不要每请求重复加载
- **验收**：前端看到的预标注是模型真输出的

### P3 —— 联调 + 提效统计
- [ ] "智能选样"按钮 → 调 `/api/sample` 拉下一批
- [ ] 提效统计面板：单张修改次数、标注量对比、时延
- [ ] 异常处理：模型超时/图片损坏/空结果都要有兜底
- **验收**：能完整演示"导入→预标→修正→选样→统计"闭环

---

## 5. 协作约定

1. **契约先行**：字段名/路径不改，要改先发群里确认。
2. **mock 先行**：不要等模型，先用假 JSON 把前后端联调通。
3. **分支**：`feature/backend`、`feature/frontend`，PR 合并到 `main`，不要直接推 main。
4. **别提交**：`node_modules/`、`dist/`、`.env`、`*.pt`、数据集。
5. **每天同步**：当前进度 + 卡点，一句话即可。
6. **时延**：服务层自身开销要小，别把 300ms 预算花在序列化上。

---

## 6. 模型侧接口约定（待实现 → 先按签名 mock）

> ⚠️ 当前 `inference/`、`active_learning/` **仅有 README，入口脚本尚未实现**。下面是已约定的签名，你先按它 mock，P2 直接替换。

```python
# 模型侧将提供（你 mock 时照这个签名）
def predict(image_path: str) -> list[Instance]:
    """返回 [{class_id, class_name, score, polygon, bbox}, ...]"""

def select(pool: list[str], k: int, strategy: str) -> list[str]:
    """strategy: random | entropy | tta | coreset；返回被选中的 image_id"""
```

**权重位置**：训练产物在 `runs/plantseg34_baseline/weights/best.pt`，**不入库**，后端启动时加载一次。

---

## 7. 你的工作对应比赛多少分（附件4 评分表）

| 维度 | 分值 | 与你的关系 |
|---|---|---|
| 人机协同原型完整度 | **25** | 预标—修正—回流闭环可演示（15）+ 提效指标量化（10）→ **主要靠你** |
| 工程实现与可复现性 | **15** | 代码质量与稳定性（8）+ 可复现文档（7）→ **你占大头** |
| 预标注质量 / 主动学习有效性 | 55 | 模型侧负责，但你实现的界面决定了它们"能不能被看见" |

**换句话说：55 分是模型算出来的，但能不能拿到这 55 分，取决于你的系统能不能把它演示出来。**

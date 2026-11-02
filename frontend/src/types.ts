/** 与后端契约一一对应。改这里之前先看 docs/INTERFACE_ADDENDUM.md。 */

export type Instance = {
  /** 前端本地 id，后端不认；提交时去掉 */
  id: string
  class_id: number
  class_name: string
  /** 模型置信度；人工新画的是 null */
  score: number | null
  /** 归一化多边形 [[x,y], ...]，0..1 */
  polygon: [number, number][]
  source: 'pred' | 'manual'
}

export type ImageItem = {
  image_id: string
  file_name: string
  split: string
  width: number
  height: number
  gt_class_id: number | null
  gt_class_name: string | null
  disease: string | null
}

export type ClassItem = {
  class_id: number
  class_name: string
}

export type PredictResponse = {
  instances: Omit<Instance, 'id' | 'source'>[]
  latency_ms: number
  image_width: number
  image_height: number
  model_version: string
}

export type HealthResponse = {
  status: 'ok'
  model: string
  classes: number
  dataset: {
    mode: 'manifest' | 'demo'
    note: string
    total: number
    splits: { name: string; count: number }[]
    classes_present: number
  }
  model_info: {
    weights: string
    model_version: string
    device: string
    imgsz: number
    conf: number
    nc: number
    load_ms: number
  }
}

export type ImageListResponse = {
  total: number
  offset: number
  limit: number
  items: ImageItem[]
}

/** 操作日志的词表 —— 与 src/m5_logstats.py 的 COUNTED / NEUTRAL 严格对齐 */
export type OpName =
  | '改类别'
  | '删实例'
  | '标为无目标'
  | '整张确认'
  | '撤销'
  | '提交'
  | '新画'
  | '拖边界'

/** 计入"单张修改次数"的操作（m5 COUNTED） */
export const COUNTED_OPS: ReadonlySet<OpName> = new Set<OpName>([
  '改类别',
  '删实例',
  '标为无目标',
])

export type Op = {
  ts: string
  /** 文件名的 stem，不是 image_id —— m5_logstats.py 按它分组 */
  image: string
  op: OpName
  instance: number
  detail: string
  n_ops_on_image: number
  api_ms: string
}

export type LabelState = 'labeled' | 'no_target' | 'unlabeled'

// ---------------------------------------------------------------- 主动学习

export type SampleStrategy = 'random' | 'entropy' | 'tta' | 'coreset'

export type SampleResponse = {
  selected: string[]
  /** rank 即选样顺序；score 恒为 null（模型侧 select() 只返回路径，不返回分值） */
  reasons: Record<string, { rank: number; criterion: string; score: number | null }>
  strategy: SampleStrategy
  pool_size: number
  k: number
  latency_ms: number
  note: string
}

// ---------------------------------------------------------------- 点提示

export type RefineResponse = {
  found: boolean
  class_id: number | null
  class_name: string | null
  score: number | null
  /** 归一化坐标；未检出时为 null —— 这是正常返回，不是错误 */
  polygon: [number, number][] | null
  bbox: number[] | null
  /** 掩码面积 / ROI 面积；> 0.85 说明模型把整个选区当成一个实例了 */
  coverage: number | null
  raw_vertices: number | null
  latency_ms: number
}

/** 画布交互模式：select 点选 / refine 点一下让模型重识别 / draw 手画多边形 */
export type CanvasMode = 'select' | 'refine' | 'draw'

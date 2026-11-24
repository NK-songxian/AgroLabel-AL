/**
 * 全局状态（Zustand）。
 *
 * 一条原则：**改类 / 删除 / 拖顶点都不调后端**，只有"提交"才落盘。
 * 这样单张图 5 次修改 = 5 次本地操作 + 1 次请求，不把网络往返摊进 300ms 预算。
 * （见 docs/INTERFACE_ADDENDUM.md §5）
 */

import { create } from 'zustand'

import { NotImplementedError, api } from './api'
import type {
  ClassItem,
  HealthResponse,
  ImageItem,
  Instance,
  LabelState,
  Arm,
  CanvasMode,
  Op,
  OpName,
  SampleStrategy,
  ExperimentInfo,
  ExpTask,
} from './types'
import { applyTheme, initialTheme, type Theme } from './theme'
import { ARM_LABEL, COUNTED_OPS, taskKey } from './types'

type Status = 'booting' | 'idle' | 'loading' | 'predicting' | 'ready' | 'error' | 'done'

type Snapshot = { instances: Instance[]; nOps: number; labelState: LabelState }

export type Toast = { id: number; kind: 'ok' | 'warn' | 'err'; text: string }

const PAGE = 60
const MAX_UNDO = 50
/** 标注员标识存在这里，刷新不丢 */
const OPERATOR_KEY = 'agrolabel_operator'

function readStoredOperator(): string {
  try {
    return localStorage.getItem(OPERATOR_KEY) ?? ''
  } catch {
    return ''
  }
}
const SESSION_ID = `s${Date.now().toString(36)}`

/** 日志上送的攒批参数：够 5 条或停手 0.8s 就冲一次，提交时强制冲 */
const FLUSH_AT = 5
const FLUSH_DELAY_MS = 800

// 这两个是运行时态，不该进 store（会触发无意义的重渲染）
let queue: Op[] = []
let flushTimer: ReturnType<typeof setTimeout> | null = null
let flushing = false

/** 打开图像的递增序号：用于丢弃快速翻页时回来的过期 predict 响应 */
let openSeq = 0

/** submit 防重入标志（双击会跳过一张图） */
let submitting = false

const uid = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2)

/** m5_logstats.py 按文件名的 stem 分组，不是 image_id */
const stemOf = (fileName: string) => fileName.replace(/\.[^.]+$/, '')

const nowTs = () => {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
    `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

type Store = {
  health: HealthResponse | null
  classes: ClassItem[]
  bootError: string | null

  queue: ImageItem[]
  queueTotal: number
  split: string
  queueLoading: boolean
  /** 队列来源，显示在左栏标题上：全部 / 智能选样 · entropy */
  /** 队列搜索关键词（空串 = 不过滤） */
  queueQuery: string
  queueLabel: string
  /** 智能选样时每张图的打分（image_id -> {score, criterion}）；普通队列为空对象 */
  sampleScores: Record<string, { score: number | null; criterion: string }>

  index: number
  current: ImageItem | null
  status: Status
  error: string | null

  instances: Instance[]
  predSnapshot: Instance[]
  selectedId: string | null
  labelState: LabelState
  nOps: number
  startedAt: number
  latencyMs: number | null
  modelVersion: string | null

  undoStack: Snapshot[]

  ops: Op[]
  submitted: Record<string, number>

  scale: number
  tx: number
  ty: number
  fitToken: number

  /** 画布模式：select 点选实例 / refine 点一下让模型重识别 */
  mode: CanvasMode
  /** 点提示请求进行中（与 predict 的 status 分开，避免文案串台） */
  refining: boolean
  /** 画布是否显示标注多边形；关掉可以干净地看原图（数据不受影响） */
  showMasks: boolean
  /** 明暗主题。默认白天，左下角可切 */
  theme: Theme
  /** 标注员标识（姓名/编号）。写进每张图的元数据，交叉分配时按人统计用 */
  operator: string
  /**
   * 这张图上"第一个标注动作"的时刻。null = 还没动过。
   *
   * 计时口径（手册 §08）：起点不是"打开图"，而是**人真正开始动手**——
   * 纯人工臂 = 画下第一个顶点；其余两臂 = 第一次点选/改类/删/拖顶点/点提示。
   * 这样排除掉"看图、找病灶、发呆"的时间，三臂比的才是标注动作本身。
   * 打开时刻仍保留在 startedAt 里，落盘时两个口径都会上报。
   */
  actionAt: number | null
  /**
   * 实验臂。决定三件事：是否跑预标注（manual 不跑）、是否允许点提示
   * （noprompt 禁用）、产物落到 out/annotations/<arm>/ 哪个目录。
   */
  arm: Arm
  /**
   * 三臂实验清单（GET /api/experiment）。null = 还没载入过。
   * 载入成功后 `experimentOn` 为真：队列换成清单里的任务，臂由任务决定。
   */
  experiment: ExperimentInfo | null
  /** 实验模式开关。开着时 setArm 会被拒，避免臂标错把数据弄废 */
  experimentOn: boolean
  /** 与 queue 下标一一对应的任务（一张图可能出现 3 次，三个臂各一次） */
  expTasks: ExpTask[]
  /** 已完成的任务，key = `image_id|arm`。本地累加，免得每提交一次就重拉清单 */
  expDone: Record<string, true>

  toasts: Toast[]

  bootstrap: () => Promise<void>
  loadQueue: (split?: string) => Promise<void>
  /** 只更新搜索词；真正的请求由面板防抖后调 loadQueue */
  setQueueQuery: (q: string) => void
  openIndex: (i: number) => Promise<void>
  reloadPredict: () => Promise<void>
  next: () => void
  prev: () => void

  select: (id: string | null) => void
  deleteSelected: () => void
  changeClass: (id: string, classId: number) => void
  markNoTarget: () => void
  confirmAll: () => void
  undo: () => void
  submit: () => Promise<void>

  setMode: (m: CanvasMode) => void
  toggleMasks: () => void
  /** 明暗切换：改 <html data-theme>，并记住选择 */
  toggleTheme: () => void
  /** 设置标注员，记住选择 */
  setOperator: (v: string) => void
  /** 记下"这张图的第一个标注动作"；已有值则不覆盖 */
  markAction: () => void
  /** 切换实验臂；会按新条件重新载入当前图 */
  setArm: (a: Arm) => void
  /** 读实验清单；成功后把队列换成清单里的图并开启实验模式 */
  loadExperiment: () => Promise<void>
  /** 退出实验模式，回到普通队列 */
  exitExperiment: () => void

  // ---- 手画多边形
  /** 正在画的多边形顶点（归一化坐标）。null 表示没在画 */
  draft: [number, number][] | null
  drawAddPoint: (x: number, y: number) => void
  drawUndoPoint: () => void
  /** 闭合：≥3 个点才成实例，返回是否成功 */
  drawCommit: () => boolean
  drawCancel: () => void

  // ---- 拖顶点
  /** 按下顶点时压一次撤销栈（整段拖动只算一次撤销） */
  beginVertexDrag: () => void
  moveVertex: (id: string, index: number, x: number, y: number) => void
  endVertexDrag: () => void
  /** 删掉某个顶点（橡皮擦模式）。少于 3 个顶点时拒绝。 */
  deleteVertex: (id: string, index: number) => boolean
  /** 给选中实例插入一个顶点（落在最长边的中点）。选中了才可用。 */
  addVertex: () => boolean
  /**
   * 在指定位置插一个顶点：找离该点最近的边，把它劈成两段。
   * 坐标用**图像像素**，内部转归一化 —— 距离必须在像素空间量，
   * 否则 302×225 这种非方形图上归一化距离会被拉伸，选错边。
   */
  insertVertexAt: (px: number, py: number, imgW: number, imgH: number) => boolean
  /** 点提示：在图像像素坐标 (px,py) 处裁 ROI 重识别，成功后插入一个新实例 */
  refineAt: (px: number, py: number) => Promise<void>
  /** 主动学习选样，把结果换成新队列 */
  loadSample: (strategy: SampleStrategy, k: number) => Promise<void>

  /** 记录一条操作日志。口径对齐 src/m4_app.py 的 log() */
  logOp: (op: OpName, detail: string, instance?: number) => void

  /** 把攒着的日志立刻上送到 /api/logs/ops。force=false 时失败不重排定时器。 */
  flushOps: (force?: boolean) => Promise<void>

  setViewport: (v: { scale?: number; tx?: number; ty?: number }) => void
  requestFit: () => void

  toast: (text: string, kind?: Toast['kind']) => void
  dismissToast: (id: number) => void
  exportOpsCsv: () => string
  countedForCurrent: () => number
}

export const useStore = create<Store>((set, get) => ({
  health: null,
  classes: [],
  bootError: null,

  queue: [],
  queueTotal: 0,
  split: 'all',
  queueQuery: '',
  queueLoading: false,
  queueLabel: '全部',
  sampleScores: {},

  index: -1,
  current: null,
  status: 'booting',
  error: null,

  instances: [],
  predSnapshot: [],
  selectedId: null,
  labelState: 'unlabeled',
  nOps: 0,
  startedAt: 0,
  latencyMs: null,
  modelVersion: null,

  undoStack: [],
  ops: [],
  submitted: {},

  scale: 1,
  tx: 0,
  ty: 0,
  fitToken: 0,

  mode: 'select',
  refining: false,
  /** 画布是否显示标注多边形。纯视图开关，不影响实例数据本身 */
  showMasks: true,
  theme: initialTheme(),
  operator: readStoredOperator(),
  actionAt: null,
  arm: 'assisted',
  experiment: null,
  experimentOn: false,
  expTasks: [],
  expDone: {},
  draft: null,

  toasts: [],

  // ------------------------------------------------------------ 启动

  bootstrap: async () => {
    set({ status: 'booting', bootError: null })
    try {
      const [health, cls] = await Promise.all([api.health(), api.classes()])
      set({ health, classes: cls.classes, status: 'idle' })
      await get().loadQueue('all')
      if (get().queue.length) await get().openIndex(0)
    } catch (e) {
      set({ status: 'error', bootError: e instanceof Error ? e.message : String(e) })
    }
  },

  loadQueue: async (split) => {
    const s = split ?? get().split
    const kw = get().queueQuery.trim()
    set({ queueLoading: true, split: s })
    try {
      const res = await api.images({
        split: s, offset: 0, limit: PAGE,
        ...(kw ? { q: kw } : {}),
      })
      set({
        queue: res.items, queueTotal: res.total, queueLoading: false,
        queueLabel: kw ? `搜索「${kw}」` : (s === 'all' ? '全部' : s),
        sampleScores: {},
      })
    } catch (e) {
      set({ queueLoading: false })
      get().toast(`队列加载失败：${e instanceof Error ? e.message : e}`, 'err')
    }
  },

  setQueueQuery: (q) => set({ queueQuery: q }),

  // ------------------------------------------------------------ 实验模式

  /**
   * 读实验清单，并把队列换成清单里的图。
   *
   * 为什么必须这么做：实验只标清单里的 68 张，而全量数据集有 7773 张；
   * 队列按字母序排，人根本找不到该标的那几张。更要紧的是**每张图绑定一个臂**——
   * 靠人记、靠手切臂，标错一次这条数据就废了（产物按臂分目录落盘）。
   */
  loadExperiment: async () => {
    let info: ExperimentInfo
    try {
      info = await api.experiment()
    } catch (e) {
      get().toast(`读实验清单失败：${e instanceof Error ? e.message : e}`, 'err')
      return
    }
    set({ experiment: info })
    if (!info.loaded) {
      get().toast(info.note || '没有可用的实验清单', 'warn')
      return
    }

    // 按任务顺序取图元信息（宽高、GT 类名）。/api/images?ids= 单次上限 200。
    // 同图三臂时一张图会出现 3 次，去重后再取。
    const uniqIds = [...new Set(info.tasks.map((t) => t.image_id))]
    const byId = new Map<string, ImageItem>()
    try {
      for (let i = 0; i < uniqIds.length; i += 200) {
        const res = await api.images({ ids: uniqIds.slice(i, i + 200), limit: 200 })
        for (const it of res.items) byId.set(it.image_id, it)
      }
    } catch (e) {
      get().toast(`载入实验图失败：${e instanceof Error ? e.message : e}`, 'err')
      return
    }

    // queue 与 expTasks 必须同下标对齐：给不存在的图建任务只会让人点开报错，
    // 所以缺图的那些任务一起丢掉，并在下面显式报数。
    const queue: ImageItem[] = []
    const tasks: ExpTask[] = []
    const missing: string[] = []
    for (const t of info.tasks) {
      const it = byId.get(t.image_id)
      if (!it) {
        if (!missing.includes(t.image_id)) missing.push(t.image_id)
        continue
      }
      queue.push(it)
      tasks.push(t)
    }

    const expDone: Record<string, true> = {}
    for (const k of info.done_tasks) expDone[k] = true

    set({
      queue,
      expTasks: tasks,
      queueTotal: queue.length,
      queueLabel: `实验 · ${info.n_images} 图 / ${queue.length} 任务`,
      experimentOn: true,
      expDone,
      index: -1,
      current: null,
      status: 'idle',
    })

    if (missing.length) {
      // 清单和数据集对不上时必须喊出来：否则队列静默变短，人以为标完了
      get().toast(
        `清单里有 ${missing.length} 张不在当前数据集（如 ${missing[0]}），已跳过`,
        'err',
      )
    } else {
      const parts = (Object.keys(ARM_LABEL) as Arm[])
        .filter((a) => (info.arms[a] ?? 0) > 0)
        .map((a) => `${ARM_LABEL[a]} ${info.arms[a]}`)
        .join(' / ')
      get().toast(`实验模式已开启：${queue.length} 张（${parts}）`, 'ok')
    }
    if (queue.length) await get().openIndex(0)
  },

  exitExperiment: () => {
    // 退出时把臂复位成 assisted：普通标注模式下要保证"预标注 + 点提示"都能用，
    // 不能因为刚才在 noprompt 臂里就把点提示一直禁着。
    set({ experimentOn: false, experiment: null, expDone: {}, expTasks: [],
          arm: 'assisted', queueQuery: '' })
    void get().loadQueue('all')
    get().toast('已退出实验模式', 'ok')
  },

  // ------------------------------------------------------------ 打开一张图

  openIndex: async (i) => {
    const { queue } = get()
    if (i < 0 || i >= queue.length) return
    const img = queue[i]

    // 实验模式：臂由**任务**决定（同图三臂时，同一张图三个任务分别是三个臂），进图即切换。
    // 这里直接 set 而不走 setArm —— setArm 会重新 openIndex，递归。
    if (get().experimentOn) {
      const want = get().expTasks[i]?.arm
      if (want && want !== get().arm) set({ arm: want, mode: 'select', draft: null })
    }

    const seq = ++openSeq

    set({
      index: i,
      current: img,
      status: 'predicting',
      error: null,
      instances: [],
      predSnapshot: [],
      selectedId: null,
      labelState: 'unlabeled',
      nOps: 0,
      undoStack: [],
      startedAt: Date.now(),
      actionAt: null,
      latencyMs: null,
      scale: 1,
      tx: 0,
      ty: 0,
      draft: null,
    })
    get().requestFit()

    try {
      // 先看这张有没有存过的标注：有就恢复，别再拿预标注覆盖人的工作。
      // （之前刷新就丢数据的根因除了 confirmAll 不落盘，还有这里永远走 predict）
      const saved = await api.getAnnotation(img.image_id, get().arm)
      if (seq !== openSeq) return
      if (saved?.found) {
        const list: Instance[] = saved.instances.map((it) => ({
          id: uid(),
          class_id: it.class_id,
          class_name: it.class_name,
          score: null,
          polygon: it.polygon,
          source: 'saved' as const,
        }))
        set({
          instances: list,
          predSnapshot: [],
          status: 'ready',
          labelState: saved.label_state,
          latencyMs: null,
          modelVersion: null,
        })
        get().toast(
          `已恢复上次保存的标注（${list.length} 个实例）`,
          'ok',
        )
        return
      }

      // 纯人工臂：读到这就说明这张没标过，**不跑预标注** —— 基线必须从空白开始，
      // 否则这一臂和 assisted 没区别，对照就是假的。
      // 但上面的"读回已保存标注"必须照做：纯人工臂的产物也是产物，
      // 刷新后同样不能丢。
      if (get().arm === 'manual') {
        set({
          status: 'ready',
          instances: [],
          predSnapshot: [],
          labelState: 'unlabeled',
          latencyMs: null,
          modelVersion: null,
        })
        return
      }

      const res = await api.predict(img.image_id)
      // 快速连按 ←/→ 时，先发的请求可能后回来 —— 用序号把过期响应丢掉，
      // 否则会把上一张图的多边形贴到当前这张上
      if (seq !== openSeq) return
      const list: Instance[] = res.instances.map((it) => ({ ...it, id: uid(), source: 'pred' }))
      set({
        instances: list,
        predSnapshot: list.map((x) => ({ ...x })),
        status: 'ready',
        labelState: list.length ? 'labeled' : 'unlabeled',
        latencyMs: res.latency_ms,
        modelVersion: res.model_version,
      })
    } catch (e) {
      if (seq !== openSeq) return
      set({ status: 'error', error: e instanceof Error ? e.message : String(e) })
      get().toast('预标注失败，可手工标注', 'warn')
    }
  },

  reloadPredict: async () => {
    const cur = get().current
    if (!cur) return
    const seq = ++openSeq
    set({ status: 'predicting' })
    try {
      const res = await api.predict(cur.image_id)
      if (seq !== openSeq || get().current?.image_id !== cur.image_id) return
      const list: Instance[] = res.instances.map((it) => ({ ...it, id: uid(), source: 'pred' }))
      set({
        instances: list,
        predSnapshot: list.map((x) => ({ ...x })),
        status: 'ready',
        latencyMs: res.latency_ms,
        modelVersion: res.model_version,
      })
    } catch (e) {
      if (seq !== openSeq) return
      set({ status: 'error' })
      get().toast(`重跑预标注失败：${e instanceof Error ? e.message : e}`, 'err')
    }
  },

  next: () => {
    const { index, queue } = get()
    if (index + 1 < queue.length) void get().openIndex(index + 1)
    else get().toast('已经是队列最后一张', 'warn')
  },

  prev: () => {
    const { index } = get()
    if (index > 0) void get().openIndex(index - 1)
    else get().toast('已经是队列第一张', 'warn')
  },

  // ------------------------------------------------------------ 编辑

  select: (id) => {
    if (id != null) get().markAction()
    set({ selectedId: id })
  },

  logOp: (op, detail, instance = -1) => {
    const { current, nOps, ops, latencyMs } = get()
    if (!current) return
    const entry: Op = {
      ts: nowTs(),
      image: stemOf(current.file_name),
      op,
      instance,
      detail,
      n_ops_on_image: nOps, // 已在上一步自增，与 m4_app 的写法一致
      api_ms: latencyMs != null ? latencyMs.toFixed(1) : '',
    }
    set({ ops: [...ops, entry] })

    // 攒批上送后端，写成 out/logs/ops.csv 供 m5_logstats.py 消费
    queue.push(entry)
    if (queue.length >= FLUSH_AT) void flushOps(false)
    else scheduleFlush()
  },

  flushOps: (force = false) => flushOps(force),

  deleteSelected: () => {
    get().markAction()
    const { selectedId, instances, undoStack, nOps, labelState } = get()
    if (!selectedId) return
    const idx = instances.findIndex((x) => x.id === selectedId)
    if (idx < 0) return
    const gone = instances[idx]

    set({
      undoStack: [...undoStack, { instances, nOps, labelState }].slice(-MAX_UNDO),
      instances: instances.filter((x) => x.id !== selectedId),
      selectedId: null,
      nOps: nOps + 1,
      labelState: 'labeled',
    })
    get().logOp('删实例', gone.class_name, idx)
  },

  changeClass: (id, classId) => {
    get().markAction()
    const { instances, classes, undoStack, nOps, labelState } = get()
    const idx = instances.findIndex((x) => x.id === id)
    if (idx < 0) return
    const target = classes.find((c) => c.class_id === classId)
    if (!target) return
    const old = instances[idx].class_name
    if (old === target.class_name) return // 没变就不计数，和 m4_app 一致

    const next = instances.slice()
    next[idx] = { ...next[idx], class_id: target.class_id, class_name: target.class_name }
    set({
      undoStack: [...undoStack, { instances, nOps, labelState }].slice(-MAX_UNDO),
      instances: next,
      nOps: nOps + 1,
      labelState: 'labeled',
    })
    get().logOp('改类别', `${old} -> ${target.class_name}`, idx)
  },

  markNoTarget: () => {
    // 快照里必须存**操作前**的 labelState。写成 'no_target' 的话，
    // 撤销后实例回来了但 label_state 还停在 no_target，
    // 提交时会被后端拒（schemas.py: no_target 且 instances 非空 -> 422），人卡在这张图上。
    const { instances, undoStack, nOps, labelState } = get()
    set({
      undoStack: [...undoStack, { instances, nOps, labelState }].slice(-MAX_UNDO),
      instances: [],
      selectedId: null,
      nOps: nOps + 1,
      labelState: 'no_target',
    })
    get().logOp('标为无目标', '')
    void persist()
  },

  confirmAll: () => {
    const { instances, undoStack, nOps, labelState, status, current } = get()
    // 没有图 / 预标注还没回来时不许"整张确认"：
    // 否则空画布被确认并 persist，落盘成一条空标注（三臂实验里就是废数据）。
    if (!current) return
    if (status === 'predicting') {
      get().toast('预标注还没回来，等结果出来再确认', 'warn')
      return
    }
    set({
      undoStack: [...undoStack, { instances, nOps, labelState }].slice(-MAX_UNDO),
      nOps: nOps + 1,
      labelState: labelState === 'unlabeled' ? 'labeled' : labelState,
    })
    get().logOp('整张确认', `${instances.length} 个实例`)
    // 「确认」在标注员心里就是"这张我认可了，存下来"。
    // 之前只写日志不落盘，刷新就全丢 —— 必须真存。
    void persist()
  },

  undo: () => {
    const { undoStack } = get()
    if (!undoStack.length) {
      get().toast('没有可撤销的操作', 'warn')
      return
    }
    const snap = undoStack[undoStack.length - 1]
    set({
      undoStack: undoStack.slice(0, -1),
      instances: snap.instances,
      nOps: snap.nOps,
      labelState: snap.labelState,
      selectedId: null,
    })
    // m5 把"撤销"归为 NEUTRAL：不计数，只抵消前一次
    get().logOp('撤销', '')
  },

  // ------------------------------------------------------------ 提交

  submit: async () => {
    // 防重入：双击「提交并下一张」会走两遍 persist + 两次 next()，
    // 结果是静默跳过一张图 —— 三臂实验里就是丢一个数据点。
    if (submitting) return
    submitting = true
    const { current, instances, labelState, arm } = get()
    if (!current) {
      submitting = false
      return
    }

    if (labelState === 'unlabeled' && instances.length === 0) {
      get().toast('这张还没标：先确认，或点"标为无目标"', 'warn')
      submitting = false
      return
    }

    const ok = await persist(`已提交 ${instances.length} 个实例，正在下一张`)
    if (!ok) {
      submitting = false
      return
    }
    get().logOp('提交', `${instances.length} 个实例`)
    set({
      submitted: { ...get().submitted, [current.image_id]: instances.length },
      status: 'done',
    })
    // 记下这个**任务**已完成（同图三臂时一张图有三个任务，不能只用 image_id 当 key）
    if (get().experimentOn) {
      set({ expDone: { ...get().expDone, [taskKey(current.image_id, arm)]: true } })
    }
    // m5 靠"提交"事件确定分母，必须在上送完再翻页
    await flushOps(true)
    // 记住提交的是哪一张：这 260ms 里人可能已经手动翻页或点了别的队列项，
    // 若那时还按当前 index 往前跳，就会从新位置再跳一张 —— 三臂实验里等于丢一个数据点。
    const submittedId = current.image_id
    setTimeout(() => {
      submitting = false
      if (get().current?.image_id === submittedId) void get().next()
    }, 260)
  },

  // ------------------------------------------------------------ 点提示

  setMode: (m) => set({ mode: m, draft: null }),

  setArm: (a) => {
    // 非实验模式（普通标注）：臂只决定"产物落哪个目录"，界面上不暴露这个选择，
    // 默认 assisted（预标注 + 点提示都可用）。
    if (!get().experimentOn) {
      if (a === get().arm) return
      set({ arm: a, mode: 'select', draft: null })
      const i = get().index
      if (i >= 0) void get().openIndex(i)
      get().toast(`已切到「${ARM_LABEL[a]}」，当前图已重新载入`, 'ok')
      return
    }
    // 实验模式：在这选臂 = **跳到该臂的下一个未完成任务**，而不是直接改臂。
    // 臂必须跟着任务走，否则产物会落到 out/annotations/<arm>/ 错误目录，
    // 而且事后没法凭数据判断谁标错了。
    const { expTasks, expDone, index } = get()
    const at = expTasks.findIndex(
      (tk, i) => tk.arm === a && i !== index && !expDone[taskKey(tk.image_id, tk.arm)],
    )
    if (at < 0) {
      get().toast(`「${ARM_LABEL[a]}」已经没有未完成的任务了`, 'warn')
      return
    }
    void get().openIndex(at)
    get().toast(`跳到「${ARM_LABEL[a]}」的下一个任务（第 ${at + 1} 个）`, 'ok')
  },

  toggleMasks: () => set((s) => ({ showMasks: !s.showMasks })),

  setOperator: (v) => {
    const name = v.slice(0, 32)
    try { localStorage.setItem(OPERATOR_KEY, name) } catch { /* 存不了也不影响本会话 */ }
    set({ operator: name })
  },

  markAction: () => {
    if (get().actionAt == null) set({ actionAt: Date.now() })
  },

  toggleTheme: () => {
    const next: Theme = get().theme === 'dark' ? 'light' : 'dark'
    applyTheme(next)
    set({ theme: next })
  },

  // ------------------------------------------------------------ 手画多边形

  drawAddPoint: (x, y) => {
    get().markAction()
    set((s) => ({ draft: [...(s.draft ?? []), [x, y] as [number, number]] }))
  },

  drawUndoPoint: () =>
    set((s) => {
      const d = s.draft ?? []
      return { draft: d.length > 1 ? d.slice(0, -1) : null }
    }),

  drawCommit: () => {
    const { draft, current, instances, undoStack, nOps, labelState, classes } = get()
    if (!draft || draft.length < 3 || !current) {
      set({ draft: null })
      return false
    }

    // 类别跟点提示保持一致：图像的 gt 类优先。人在病斑上描的是"这张图的病"，
    // 逐点选类别太慢，改类随时可以在右栏做。
    const cid = current.gt_class_id ?? 0
    const cname = current.gt_class_name
      ?? classes.find((c) => c.class_id === cid)?.class_name
      ?? ''

    const inst: Instance = {
      id: uid(),
      class_id: cid,
      class_name: cname,
      score: null,
      polygon: draft,
      source: 'manual',
    }

    set({
      undoStack: [...undoStack, { instances, nOps, labelState }].slice(-MAX_UNDO),
      instances: [...instances, inst],
      selectedId: inst.id,
      labelState: 'labeled',
      draft: null,
    })
    get().logOp('新画', `手画 ${cname}（${draft.length} 顶点）`, instances.length)
    return true
  },

  drawCancel: () => set({ draft: null }),

  // ------------------------------------------------------------ 拖顶点

  beginVertexDrag: () => {
    get().markAction()
    const { instances, undoStack, nOps, labelState } = get()
    // 整段拖动只压一次，否则每帧一次撤销，Ctrl+Z 要按几百下
    set({
      undoStack: [...undoStack, { instances, nOps, labelState }].slice(-MAX_UNDO),
    })
  },

  moveVertex: (id, index, x, y) =>
    set((s) => ({
      instances: s.instances.map((it) =>
        it.id !== id
          ? it
          : {
              ...it,
              polygon: it.polygon.map((p, i) =>
                i === index ? ([x, y] as [number, number]) : p,
              ),
            },
      ),
    })),

  endVertexDrag: () => {
    const { instances, selectedId } = get()
    const idx = instances.findIndex((i) => i.id === selectedId)
    const sel = idx >= 0 ? instances[idx] : null
    get().logOp(
      '拖边界',
      sel ? `${sel.class_name}（${sel.polygon.length} 顶点）` : '顶点微调',
      idx, // instance 列是下标，不是总数
    )
  },

  deleteVertex: (id, index) => {
    get().markAction()
    const { instances, undoStack, nOps, labelState } = get()
    const sel = instances.find((i) => i.id === id)
    if (!sel) return false
    if (sel.polygon.length <= 3) {
      // 两个点的"多边形"没有面积，m6 会把退化的实例直接丢掉，统计就对不上了
      get().toast('只剩 3 个顶点了，再删就不成多边形；整个实例请用「删实例」', 'warn')
      return false
    }
    const polygon = sel.polygon.filter((_, i) => i !== index)
    set({
      undoStack: [...undoStack, { instances, nOps, labelState }].slice(-MAX_UNDO),
      instances: instances.map((it) => (it.id === id ? { ...it, polygon } : it)),
      selectedId: id,
    })
    get().logOp(
      '拖边界',
      `${sel.class_name} 删节点（${polygon.length} 顶点）`,
      instances.findIndex((i) => i.id === id),
    )
    return true
  },

  addVertex: () => {
    get().markAction()
    const { instances, selectedId, undoStack, nOps, labelState } = get()
    const sel = instances.find((i) => i.id === selectedId)
    if (!sel || sel.polygon.length < 2) return false

    // 插在最长边的中点：那里通常是多边形最粗糙、最需要补点的地方。
    // 加完新顶点就是选中态的白色圆点，直接拖到准确位置即可。
    let edge = 0
    let longest = -1
    for (let i = 0; i < sel.polygon.length; i++) {
      const a = sel.polygon[i]
      const b = sel.polygon[(i + 1) % sel.polygon.length]
      const len = Math.hypot(b[0] - a[0], b[1] - a[1])
      if (len > longest) {
        longest = len
        edge = i
      }
    }
    const a = sel.polygon[edge]
    const b = sel.polygon[(edge + 1) % sel.polygon.length]
    const mid: [number, number] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
    const polygon = [
      ...sel.polygon.slice(0, edge + 1),
      mid,
      ...sel.polygon.slice(edge + 1),
    ]

    set({
      undoStack: [...undoStack, { instances, nOps, labelState }].slice(-MAX_UNDO),
      instances: instances.map((it) => (it.id === sel.id ? { ...it, polygon } : it)),
    })
    get().logOp(
      '拖边界',
      `${sel.class_name} 加节点（${polygon.length} 顶点）`,
      instances.findIndex((i) => i.id === sel.id), // 下标，不是总数
    )
    return true
  },

  insertVertexAt: (px, py, imgW, imgH) => {
    get().markAction()
    const { instances, selectedId, undoStack, nOps, labelState } = get()
    const sel = instances.find((i) => i.id === selectedId)
    if (!sel || sel.polygon.length < 3 || !imgW || !imgH) return false

    // 点到线段的最短距离（像素空间）。夹紧到 [0,1] 再量，
    // 否则点在线段延长线上时会误判成"很近"。
    const segDist = (ax: number, ay: number, bx: number, by: number) => {
      const dx = bx - ax
      const dy = by - ay
      const len2 = dx * dx + dy * dy
      if (len2 === 0) return Math.hypot(px - ax, py - ay)
      let t = ((px - ax) * dx + (py - ay) * dy) / len2
      t = Math.max(0, Math.min(1, t))
      return Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
    }

    let edge = 0
    let best = Infinity
    for (let i = 0; i < sel.polygon.length; i++) {
      const a = sel.polygon[i]
      const b = sel.polygon[(i + 1) % sel.polygon.length]
      const d = segDist(a[0] * imgW, a[1] * imgH, b[0] * imgW, b[1] * imgH)
      if (d < best) {
        best = d
        edge = i
      }
    }

    // 顶点落在鼠标位置，夹到图内
    const nx = Math.min(1, Math.max(0, px / imgW))
    const ny = Math.min(1, Math.max(0, py / imgH))
    const polygon: [number, number][] = [
      ...sel.polygon.slice(0, edge + 1),
      [nx, ny],
      ...sel.polygon.slice(edge + 1),
    ]

    set({
      undoStack: [...undoStack, { instances, nOps, labelState }].slice(-MAX_UNDO),
      instances: instances.map((it) => (it.id === sel.id ? { ...it, polygon } : it)),
    })
    get().logOp(
      '拖边界',
      `${sel.class_name} 右键插节点（${polygon.length} 顶点）`,
      instances.findIndex((i) => i.id === sel.id),
    )
    return true
  },

  refineAt: async (px, py) => {
    get().markAction()
    const { current, instances, selectedId, classes, arm } = get()
    if (!current || get().refining) return
    if (arm === 'noprompt') {
      get().toast('这一臂（预标·禁用点提示）不允许点提示', 'warn')
      return
    }

    // 类别必须钉死：refine 的 class_id 是硬过滤。
    // 实测不传的话，同一张甘蓝图会返回 Wheat（score 0.34）—— 放大到小 ROI 后
    // 模型丢掉了宿主上下文。优先用当前选中实例的类，其次图像的 gt 类。
    const sel = instances.find((i) => i.id === selectedId)
    const classId = sel?.class_id ?? current.gt_class_id ?? undefined

    set({ refining: true })
    const reqImageId = current.image_id
    try {
      const r = await api.refine({
        image_id: current.image_id,
        point: [Math.round(px), Math.round(py)],
        class_id: classId ?? undefined,
      })

      // 请求期间可能已经翻到别的图；那样这条多边形属于上一张，不能插进来
      if (get().current?.image_id !== reqImageId) return

      if (!r.found || !r.polygon || r.polygon.length < 3) {
        get().toast('这个位置没认出来，请改用手画', 'warn')
        return
      }
      if ((r.coverage ?? 0) > 0.85) {
        get().toast('选区太大，模型把整块当成一个实例了；换个位置或缩小半径', 'warn')
      }

      const cid = r.class_id ?? classId ?? 0
      const cname = r.class_name ?? classes.find((c) => c.class_id === cid)?.class_name ?? ''
      // 放大后模型常判错物种。后端会退回"不限定类别"再找一次，并标出类别不一致；
      // 这里明说一句，否则用户会以为自己点错了地方。
      if (r.class_mismatch) {
        get().toast(`识别为 ${cname}（与本图类别不同），已插入，可在右栏改类`, 'warn')
      }
      const inst: Instance = {
        id: uid(),
        class_id: cid,
        class_name: cname,
        score: r.score,
        polygon: r.polygon,
        source: 'manual',
      }

      // 重新取状态：请求飞行期间用户可能删过/改过实例，用一开始捕获的
      // instances 会把那些操作悄悄回滚掉
      const now = get()
      set({
        undoStack: [...now.undoStack,
                    { instances: now.instances, nOps: now.nOps, labelState: now.labelState }
                   ].slice(-MAX_UNDO),
        instances: [...now.instances, inst],
        selectedId: inst.id,
        labelState: 'labeled',
      })
      // 新画不计入 m5 的修改次数，但要进日志 —— m6 的 breakdown 与"模型漏检率"都要用它
      get().logOp(
        '新画',
        `点提示 ${cname}（coverage ${(r.coverage ?? 0).toFixed(2)}，${r.latency_ms.toFixed(0)}ms）`,
        now.instances.length,
      )
    } catch (e) {
      if (e instanceof NotImplementedError) {
        get().toast('后端 /api/refine 尚未实现', 'warn')
      } else {
        get().toast(`点提示失败：${e instanceof Error ? e.message : String(e)}`, 'err')
      }
    } finally {
      set({ refining: false })
    }
  },

  // ------------------------------------------------------------ 智能选样

  loadSample: async (strategy, k) => {
    set({ queueLoading: true })
    try {
      // 实验模式下把候选池限定为清单里的图：否则默认池是"整个 catalog 减已标注"，
      // 全量数据集 7772 张会撞上 /api/sample 的单次上限 500（返回 400），
      // 而且"最值得标"跨到实验外的图也没有意义。
      const expIds = get().experimentOn
        ? (get().experiment?.images ?? []).map((x) => x.image_id)
        : undefined
      const r = await api.sample(
        { k, strategy, ...(expIds?.length ? { pool: expIds } : {}) },
        get().arm,
      )
      if (!r.selected.length) {
        set({ queueLoading: false })
        get().toast('没选出新图：可能都已经标过了', 'warn')
        return
      }
      // 后端只回 id，元信息要另取一次；按 selected 的顺序还原（rank 有意义）
      const page = await api.images({ ids: r.selected })
      const byId = new Map(page.items.map((i) => [i.image_id, i]))
      const items = r.selected
        .map((id) => byId.get(id))
        .filter((x): x is ImageItem => x !== undefined)

      // reasons 里的 score 是模型侧真实打分（random 恒为 null），随队列显示
      const sampleScores: Record<string, { score: number | null; criterion: string }> = {}
      for (const id of r.selected) {
        const rs = r.reasons[id]
        if (rs) sampleScores[id] = { score: rs.score, criterion: rs.criterion }
      }
      set({
        queue: items,
        queueTotal: items.length,
        queueLoading: false,
        queueLabel: `选样 · ${strategy}`,
        sampleScores,
      })
      get().toast(
        `选样完成：${items.length} / ${r.pool_size} 张（${strategy}，${r.latency_ms.toFixed(0)}ms）`,
        'ok',
      )
      await get().openIndex(0)
    } catch (e) {
      set({ queueLoading: false })
      if (e instanceof NotImplementedError) {
        get().toast('后端 /api/sample 尚未实现', 'warn')
      } else {
        get().toast(`选样失败：${e instanceof Error ? e.message : String(e)}`, 'err')
      }
    }
  },

  // ------------------------------------------------------------ 视口

  setViewport: (v) => set(v),
  requestFit: () => set((s) => ({ fitToken: s.fitToken + 1 })),

  // ------------------------------------------------------------ UI

  toast: (text, kind = 'ok') => {
    const id = Date.now() + Math.random()
    set((s) => ({ toasts: [...s.toasts, { id, kind, text }] }))
    setTimeout(() => get().dismissToast(id), 3200)
  },

  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  /** 导出 ops.csv 文本：表头必须与 src/m4_app.py:131 完全一致 */
  exportOpsCsv: () => {
    const head = 'ts,image,op,instance,detail,n_ops_on_image,api_ms'
    const esc = (v: string | number) => {
      const s = String(v)
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
    }
    const rows = get().ops.map((o) =>
      [o.ts, o.image, o.op, o.instance, o.detail, o.n_ops_on_image, o.api_ms].map(esc).join(','),
    )
    return [head, ...rows].join('\n')
  },

  countedForCurrent: () => {
    const { ops, current } = get()
    if (!current) return 0
    const s = stemOf(current.file_name)
    return ops.filter((o) => o.image === s && COUNTED_OPS.has(o.op)).length
  },
}))

// ---------------------------------------------------------------- 日志上送

let failures = 0
const MAX_FAILURES = 3

function scheduleFlush() {
  if (flushTimer != null) return
  flushTimer = setTimeout(() => {
    flushTimer = null
    void flushOps(false)
  }, FLUSH_DELAY_MS)
}

/**
 * 把攒着的日志 POST 到 /api/logs/ops，由后端追加进 out/logs/ops.csv。
 *
 * 失败时的态度是**不打断标注**：事件继续留在内存队列里等下次，
 * 本地 `ops` 数组始终完整（右栏可随时导出 CSV），所以最坏情况只是"没写进后端文件"。
 * 连续失败 3 次就放弃上送，避免后端没起来时每个操作都打一次无效请求。
 */
async function flushOps(force: boolean): Promise<void> {
  if (flushing || queue.length === 0) return
  if (failures >= MAX_FAILURES) return

  flushing = true
  if (flushTimer != null) {
    clearTimeout(flushTimer)
    flushTimer = null
  }

  const batch = queue.slice(0, 200)
  try {
    await api.pushOps(SESSION_ID, batch, useStore.getState().arm)
    queue = queue.slice(batch.length)
    failures = 0
    if (queue.length) scheduleFlush()
  } catch {
    failures += 1
    if (failures >= MAX_FAILURES) {
      queue = []
      useStore.getState().toast('日志上送失败，已停用自动上报；仍可手动导出 CSV', 'warn')
    } else if (force) {
      scheduleFlush()
    }
  } finally {
    flushing = false
  }
}

/**
 * 把当前图的标注写进后端（out/annotations/<arm>/）。
 *
 * 抽出来是因为「整张确认」「标为无目标」「提交并下一张」都要落盘 ——
 * 之前只有 submit 落盘，标注员点「确认」以为存了，刷新就全丢。
 */
async function persist(text?: string): Promise<boolean> {
  const { current, instances, labelState, startedAt, actionAt, arm, operator } =
    useStore.getState()
  if (!current) return false

  const now = Date.now()
  // 两个口径都上报，不丢信息：
  //   elapsed_ms            = 第一个标注动作 -> 落盘（手册 §08 用这个）
  //   elapsed_from_open_ms  = 打开这张图 -> 落盘（含看图/熟悉，备用口径）
  // 若整张都没动过手（例如看了一眼就「整张确认」），退回打开时刻，
  // 否则会算出 0，把"快速确认"记成零成本。
  const from = actionAt ?? startedAt

  const payload = {
    image_id: current.image_id,
    label_state: (labelState === 'unlabeled' ? 'labeled' : labelState) as LabelState,
    instances: instances.map((x) => ({ class_id: x.class_id, polygon: x.polygon })),
    elapsed_ms: now - from,
    elapsed_from_open_ms: now - startedAt,
    operator: operator || undefined,
    session_id: SESSION_ID,
  }

  const toast = useStore.getState().toast
  try {
    await api.submitAnnotation(payload, arm)
    toast(text ?? `已保存 ${instances.length} 个实例`, 'ok')
    void flushOps(true)
    return true
  } catch (e) {
    if (e instanceof NotImplementedError) {
      toast('后端 /api/annotations 尚未实现，本条仅本地暂存', 'warn')
      return false
    }
    // 后端因为几何/类别校验打回时，把原因原样带出来，别只说"保存失败"
    toast(`保存失败：${e instanceof Error ? e.message : String(e)}`, 'err')
    return false
  }
}

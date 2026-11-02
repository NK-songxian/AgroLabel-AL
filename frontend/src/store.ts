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
  Op,
  OpName,
} from './types'
import { COUNTED_OPS } from './types'

type Status = 'booting' | 'idle' | 'loading' | 'predicting' | 'ready' | 'error' | 'done'

type Snapshot = { instances: Instance[]; nOps: number; labelState: LabelState }

export type Toast = { id: number; kind: 'ok' | 'warn' | 'err'; text: string }

const PAGE = 60
const MAX_UNDO = 50
const SESSION_ID = `s${Date.now().toString(36)}`

/** 日志上送的攒批参数：够 5 条或停手 0.8s 就冲一次，提交时强制冲 */
const FLUSH_AT = 5
const FLUSH_DELAY_MS = 800

// 这两个是运行时态，不该进 store（会触发无意义的重渲染）
let queue: Op[] = []
let flushTimer: ReturnType<typeof setTimeout> | null = null
let flushing = false

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

  toasts: Toast[]

  bootstrap: () => Promise<void>
  loadQueue: (split?: string) => Promise<void>
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
  queueLoading: false,

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
    set({ queueLoading: true, split: s })
    try {
      const res = await api.images({ split: s, offset: 0, limit: PAGE })
      set({ queue: res.items, queueTotal: res.total, queueLoading: false })
    } catch (e) {
      set({ queueLoading: false })
      get().toast(`队列加载失败：${e instanceof Error ? e.message : e}`, 'err')
    }
  },

  // ------------------------------------------------------------ 打开一张图

  openIndex: async (i) => {
    const { queue } = get()
    if (i < 0 || i >= queue.length) return
    const img = queue[i]

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
      latencyMs: null,
      scale: 1,
      tx: 0,
      ty: 0,
    })
    get().requestFit()

    try {
      const res = await api.predict(img.image_id)
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
      set({ status: 'error', error: e instanceof Error ? e.message : String(e) })
      get().toast('预标注失败，可手工标注', 'warn')
    }
  },

  reloadPredict: async () => {
    const cur = get().current
    if (!cur) return
    set({ status: 'predicting' })
    try {
      const res = await api.predict(cur.image_id)
      const list: Instance[] = res.instances.map((it) => ({ ...it, id: uid(), source: 'pred' }))
      set({
        instances: list,
        predSnapshot: list.map((x) => ({ ...x })),
        status: 'ready',
        latencyMs: res.latency_ms,
        modelVersion: res.model_version,
      })
    } catch (e) {
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

  select: (id) => set({ selectedId: id }),

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
    const { instances, undoStack, nOps } = get()
    set({
      undoStack: [...undoStack, { instances, nOps, labelState: 'no_target' as LabelState }].slice(-MAX_UNDO),
      instances: [],
      selectedId: null,
      nOps: nOps + 1,
      labelState: 'no_target',
    })
    get().logOp('标为无目标', '')
  },

  confirmAll: () => {
    const { instances, undoStack, nOps, labelState } = get()
    set({
      undoStack: [...undoStack, { instances, nOps, labelState }].slice(-MAX_UNDO),
      nOps: nOps + 1,
      labelState: labelState === 'unlabeled' ? 'labeled' : labelState,
    })
    get().logOp('整张确认', `${instances.length} 个实例`)
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
    const { current, instances, labelState, startedAt } = get()
    if (!current) return

    if (labelState === 'unlabeled' && instances.length === 0) {
      get().toast('这张还没标：先确认，或点"标为无目标"', 'warn')
      return
    }

    const payload = {
      image_id: current.image_id,
      label_state: (labelState === 'unlabeled' ? 'labeled' : labelState) as LabelState,
      instances: instances.map((x) => ({ class_id: x.class_id, polygon: x.polygon })),
      elapsed_ms: Date.now() - startedAt,
      session_id: SESSION_ID,
    }

    const finish = () => {
      get().logOp('提交', `${instances.length} 个实例`)
      set({ submitted: { ...get().submitted, [current.image_id]: instances.length }, status: 'done' })
    }

    try {
      await api.submitAnnotation(payload)
      finish()
      // m5 靠"提交"事件确定分母，必须在上送完再翻页
      await flushOps(true)
      get().toast(`已提交 ${instances.length} 个实例`, 'ok')
      setTimeout(() => void get().next(), 260)
    } catch (e) {
      if (e instanceof NotImplementedError) {
        finish()
        void flushOps(true)
        get().toast('后端 /api/annotations 尚未实现，本条仅本地暂存', 'warn')
        setTimeout(() => void get().next(), 260)
        return
      }
      const msg = e instanceof Error ? e.message : String(e)
      // 后端因为几何/类别校验打回时，把原因原样带出来，别只说"提交失败"
      get().toast(`提交失败：${msg}`, 'err')
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
    await api.pushOps(SESSION_ID, batch)
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

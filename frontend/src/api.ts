/**
 * API 客户端。
 *
 * 一律用相对路径 /api/...，由 vite.config.ts 的 proxy 转到后端 8000。
 * 生产部署时把 frontend/dist 挂到反向代理后面即可，不需要改这里。
 */

import type {
  ClassItem,
  HealthResponse,
  ImageItem,
  ImageListResponse,
  LabelState,
  Op,
  AnnotationRead,
  PredictResponse,
  RefineResponse,
  SampleResponse,
} from './types'

const BASE = import.meta.env.VITE_API_BASE ?? ''

/** 后端尚未实现的接口，抛这个便于 UI 区分"没做"和"出错" */
export class NotImplementedError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: {
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })

  if (res.status === 404 || res.status === 501) {
    // 后端没实现的接口先按"未实现"处理，不要当崩溃
    const text = await res.text().catch(() => '')
    if (text.includes('未实现') || res.status === 501) {
      throw new NotImplementedError(text || '接口未实现')
    }
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}${text ? ` — ${text.slice(0, 200)}` : ''}`)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  health: () => request<HealthResponse>('/api/health'),

  classes: () => request<{ nc: number; classes: ClassItem[] }>('/api/classes'),

  images: (params: {
    split?: string
    class_id?: number
    offset?: number
    limit?: number
    ids?: string[]
  } = {}) => {
    const sp = new URLSearchParams()
    if (params.split && params.split !== 'all') sp.set('split', params.split)
    if (params.class_id != null) sp.set('class_id', String(params.class_id))
    if (params.offset != null) sp.set('offset', String(params.offset))
    if (params.limit != null) sp.set('limit', String(params.limit))
    if (params.ids?.length) sp.set('ids', params.ids.join(','))
    const qs = sp.toString()
    return request<ImageListResponse>(`/api/images${qs ? `?${qs}` : ''}`)
  },

  predict: (image_id: string) =>
    request<PredictResponse>('/api/predict', {
      method: 'POST',
      body: JSON.stringify({ image_id }),
    }),

  /**
   * 提交标注。后端尚未实现（P1 后半段），失败时由 UI 降级为"本地暂存"。
   * 注意：pred_snapshot 不传 —— 契约 §4.2 决定由后端从 out/pred_cache 读。
   */
  submitAnnotation: (payload: {
    image_id: string
    label_state: LabelState
    instances: { class_id: number; polygon: [number, number][] }[]
    elapsed_ms: number
    operator?: string
    session_id?: string
  }) =>
    request<{ ok: boolean; image_id: string; n_instances: number; pred_cached: boolean }>(
      '/api/annotations',
      { method: 'POST', body: JSON.stringify(payload) },
    ),

  /** 上报操作日志。后端追加进 out/logs/ops.csv，供 m5_logstats.py 消费。 */
  pushOps: (session_id: string, events: Op[]) =>
    request<{ ok: boolean; appended: number; rows_total: number }>('/api/logs/ops', {
      method: 'POST',
      body: JSON.stringify({ session_id, events }),
    }),

  /**
   * 主动学习选样。pool 省略时后端用整个 catalog 并剔除已标注过的图。
   * 注意：模型侧 select() 只返回路径不返回分值，所以 reasons.score 恒为 null。
   */
  sample: (params: {
    pool?: string[]
    k?: number
    strategy?: 'random' | 'entropy' | 'tta' | 'coreset'
    exclude_annotated?: boolean
    seed?: number
  } = {}) =>
    request<SampleResponse>('/api/sample', {
      method: 'POST',
      body: JSON.stringify({
        k: 10,
        strategy: 'entropy',
        exclude_annotated: true,
        ...params,
      }),
    }),

  /**
   * 点提示：在一次点击的位置裁 ROI 放大重识别。
   *
   * ⚠️ `class_id` 一定要传：它是**硬过滤**。实测同一张甘蓝图，不传会返回 Wheat
   * （confidence 0.34），传了正确的 Cabbage 才回本类。放大到小 ROI 后模型会丢掉
   * 宿主上下文，靠调用方把类别钉住。
   */
  refine: (params: {
    image_id: string
    point: [number, number]
    class_id?: number
    radius?: number
    max_vertices?: number
  }) =>
    request<RefineResponse>('/api/refine', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  /**
   * 读回已保存的标注。没有就返回 null（**不抛错**）——
   * 调用方用它决定"恢复上次的标注"还是"重新预标注"。
   */
  getAnnotation: async (image_id: string, arm?: string): Promise<AnnotationRead | null> => {
    const q = arm ? `?arm=${encodeURIComponent(arm)}` : ''
    const res = await fetch(
      `${BASE}/api/annotations/${encodeURIComponent(image_id)}${q}`,
      { headers: { Accept: 'application/json' } },
    )
    if (res.status === 404) return null
    if (!res.ok) throw new Error(`读取标注失败：${res.status} ${await res.text()}`)
    return (await res.json()) as AnnotationRead
  },

  imageUrl: (image: ImageItem | string, bust?: number) => {
    const id = typeof image === 'string' ? image : image.image_id
    const q = bust ? `?v=${bust}` : ''
    return `${BASE}/api/images/${encodeURIComponent(id)}/file${q}`
  },
}

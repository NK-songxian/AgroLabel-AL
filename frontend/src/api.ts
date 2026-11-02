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
  PredictResponse,
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

  /** 上报操作日志。后端尚未实现（P3），失败不阻塞交互。 */
  pushOps: (session_id: string, events: Op[]) =>
    request<{ ok: boolean; appended: number }>('/api/logs/ops', {
      method: 'POST',
      body: JSON.stringify({ session_id, events }),
    }),

  imageUrl: (image: ImageItem | string, bust?: number) => {
    const id = typeof image === 'string' ? image : image.image_id
    const q = bust ? `?v=${bust}` : ''
    return `${BASE}/api/images/${encodeURIComponent(id)}/file${q}`
  },
}

import { api } from '../api'
import { useStore } from '../store'

export function TopBar() {
  const health = useStore((s) => s.health)
  const index = useStore((s) => s.index)
  const queue = useStore((s) => s.queue)
  const latencyMs = useStore((s) => s.latencyMs)
  const prev = useStore((s) => s.prev)
  const next = useStore((s) => s.next)
  const requestFit = useStore((s) => s.requestFit)
  const scale = useStore((s) => s.scale)

  const total = useStore((s) => s.queueTotal)
  const ds = health?.dataset
  const mi = health?.model_info

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" />
        <div>
          <div className="brand-name">AgroLabel-AL</div>
          <div className="brand-sub">人机协同预标注 · 赛题一</div>
        </div>
      </div>

      <div className="topbar-mid">
        {ds && (
          <span className={`chip ${ds.mode === 'demo' ? 'chip-warn' : 'chip-ok'}`} title={ds.note}>
            {ds.mode === 'demo' ? 'DEMO 数据' : '正式数据集'} · {total} 张
          </span>
        )}
        {mi && (
          <span className="chip chip-plain" title={mi.model_version}>
            {mi.device === 'cpu' ? 'CPU' : `GPU${mi.device}`} · {mi.imgsz}px
          </span>
        )}
        {latencyMs != null && (
          <span className={`chip ${latencyMs <= 300 ? 'chip-ok' : 'chip-warn'}`}>
            预标注 {latencyMs.toFixed(1)} ms
          </span>
        )}
      </div>

      <div className="topbar-right">
        <div className="pager">
          <button onClick={prev} title="上一张 (←)">‹</button>
          <span className="pager-text">
            {index >= 0 ? index + 1 : '–'} / {queue.length || '–'}
          </span>
          <button onClick={next} title="下一张 (→)">›</button>
        </div>
        <button className="ghost" onClick={requestFit} title="自适应 (F)">
          {Math.round(scale * 100)}%
        </button>
        <a className="ghost" href={api.imageUrl(queue[index]?.image_id ?? '__none__')} target="_blank" rel="noreferrer">
          原图
        </a>
      </div>
    </header>
  )
}

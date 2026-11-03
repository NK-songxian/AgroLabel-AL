import { useMemo, useState } from 'react'

import { classSolid } from '../colors'
import { useStore } from '../store'
import type { SampleStrategy } from '../types'

const STRATEGY_LABEL: Record<SampleStrategy, string> = {
  entropy: '熵 · 最不确定',
  tta: 'TTA · 扰动不一致',
  coreset: '核心集 · 最不像',
  random: '随机 · 对照',
}

/** 队列项分数徽标前缀（与后端 reasons.criterion 对应） */
const SCORE_TAG: Record<string, string> = {
  entropy: '熵',
  tta: 'TTA',
  coreset: 'div',
}

export function QueuePanel() {
  const queue = useStore((s) => s.queue)
  const index = useStore((s) => s.index)
  const queueTotal = useStore((s) => s.queueTotal)
  const queueLoading = useStore((s) => s.queueLoading)
  const queueLabel = useStore((s) => s.queueLabel)
  const split = useStore((s) => s.split)
  const submitted = useStore((s) => s.submitted)
  const openIndex = useStore((s) => s.openIndex)
  const loadQueue = useStore((s) => s.loadQueue)
  const loadSample = useStore((s) => s.loadSample)
  const classes = useStore((s) => s.classes)
  const sampleScores = useStore((s) => s.sampleScores)
  const [strategy, setStrategy] = useState<SampleStrategy>('entropy')

  const clsName = useMemo(() => {
    const m = new Map<number, string>()
    for (const c of classes) m.set(c.class_id, c.class_name)
    return m
  }, [classes])

  const splits = useStore((s) => s.health?.dataset.splits ?? [])

  return (
    <aside className="pane pane-left">
      <div className="pane-head">
        <span>
          标注队列
          <span className="pane-sublabel">{queueLabel}</span>
        </span>
        <span className="pane-count">{queueTotal}</span>
      </div>

      {splits.length > 1 && (
        <div className="split-tabs">
          {[{ name: 'all', count: queueTotal }, ...splits].map((s) => (
            <button
              key={s.name}
              className={split === s.name ? 'on' : ''}
              onClick={() => void loadQueue(s.name)}
            >
              {s.name}
            </button>
          ))}
        </div>
      )}

      <div className="sample-bar">
        <select
          value={strategy}
          onChange={(e) => setStrategy(e.target.value as SampleStrategy)}
          disabled={queueLoading}
          title="主动学习选样策略"
        >
          {(Object.keys(STRATEGY_LABEL) as SampleStrategy[]).map((s) => (
            <option key={s} value={s}>
              {STRATEGY_LABEL[s]}
            </option>
          ))}
        </select>
        <button
          onClick={() => void loadSample(strategy, 12)}
          disabled={queueLoading}
          title="调 /api/sample，把这批里最值得标的 12 张换进队列"
        >
          智能选样
        </button>
      </div>

      <div className="queue">
        {queueLoading && <div className="queue-empty">加载中…</div>}
        {!queueLoading && queue.length === 0 && (
          <div className="queue-empty">队列为空，检查后端 /api/images</div>
        )}
        {queue.map((it, i) => {
          const active = i === index
          const done = it.image_id in submitted
          return (
            <button
              key={it.image_id}
              className={`queue-item${active ? ' on' : ''}${done ? ' done' : ''}`}
              onClick={() => void openIndex(i)}
              title={`${it.image_id} · ${it.width}×${it.height}`}
            >
              <span className="queue-idx">{String(i + 1).padStart(2, '0')}</span>
              {it.gt_class_id != null && (
                <span className="dot" style={{ background: classSolid(it.gt_class_id) }} />
              )}
              <span className="queue-name">{it.file_name}</span>
              {sampleScores[it.image_id]?.score != null && (
                <span
                  className="queue-score"
                  title={`${sampleScores[it.image_id].criterion} 打分（越高越值得标）`}
                >
                  {SCORE_TAG[sampleScores[it.image_id].criterion] ?? ''}
                  {' '}
                  {sampleScores[it.image_id].score!.toFixed(2)}
                </span>
              )}
              <span className="queue-cls">{clsName.get(it.gt_class_id ?? -1) ?? '—'}</span>
              {done && <span className="queue-tick">✓</span>}
            </button>
          )
        })}
      </div>

      {/* 左下角：明暗切换 + 队列说明 */}
      <div className="pane-foot queue-foot">
        <ThemeToggle />
        <span className="foot-note">
          仅显示前 {queue.length} 张（共 {queueTotal}）
        </span>
      </div>
    </aside>
  )
}

/** 明暗切换。放在页面左下角，深色是默认（标注时长时间盯画布）。 */
function ThemeToggle() {
  const theme = useStore((s) => s.theme)
  const toggleTheme = useStore((s) => s.toggleTheme)
  const toLight = theme === 'dark'

  return (
    <button
      className="theme-toggle"
      onClick={toggleTheme}
      title={
        toLight
          ? '切到白天模式（浅色底，强光或投影时更清楚）'
          : '切到夜间模式（深色底，长时间标注更省眼）'
      }
      aria-label={toLight ? '切换到白天模式' : '切换到夜间模式'}
    >
      <span className="glyph" aria-hidden="true">
        {toLight ? '☀' : '☾'}
      </span>
      {toLight ? '白天' : '夜间'}
    </button>
  )
}

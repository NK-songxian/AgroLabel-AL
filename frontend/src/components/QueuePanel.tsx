import { useEffect, useMemo, useRef, useState } from 'react'

import { classSolid } from '../colors'
import { useStore } from '../store'
import { ARM_LABEL, taskKey, type Arm, type SampleStrategy } from '../types'
import { ExperimentPanel } from './ExperimentPanel'

/** 下拉框里的选项：真实策略 + 一个"不选样"的「全部」，方便直接浏览整队列 */
type StrategyChoice = SampleStrategy | 'all'

const STRATEGY_LABEL: Record<StrategyChoice, string> = {
  all: '全部 · 不选样',
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

/** 选样张数：默认取整池（比池子大就会返回整池，按难度排好序） */
const DEFAULT_SAMPLE_K = 50

const clampK = (v: number) => Math.max(1, Math.min(200, Math.round(v) || 1))

/** 队列行里的臂简称：一行放不下全名 */
const ARM_SHORT: Record<Arm, string> = {
  assisted: '预',
  manual: '手',
  noprompt: '禁',
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
  const queueHasMore = useStore((s) => s.queueHasMore)
  const loadMoreQueue = useStore((s) => s.loadMoreQueue)
  const loadSample = useStore((s) => s.loadSample)
  const classes = useStore((s) => s.classes)
  const sampleScores = useStore((s) => s.sampleScores)
  const experimentOn = useStore((s) => s.experimentOn)
  const expTasks = useStore((s) => s.expTasks)
  const expDone = useStore((s) => s.expDone)
  const queueQuery = useStore((s) => s.queueQuery)
  const setQueueQuery = useStore((s) => s.setQueueQuery)
  const [strategy, setStrategy] = useState<StrategyChoice>('all')
  const [sampleK, setSampleK] = useState(DEFAULT_SAMPLE_K)
  // 输入框直接受控于 store（queueQuery），不在组件里镜像一份 ——
  // 镜像就要用 effect 同步，会触发级联渲染（oxlint 的 set-state-in-effect）。
  // 防抖只作用在**请求**上：输入停顿 250ms 才打后端，否则每敲一个字查一次 7773 张。
  const firstQuery = useRef(true)
  useEffect(() => {
    if (firstQuery.current) {
      firstQuery.current = false      // 首屏由 bootstrap 负责加载，别再查一次
      return
    }
    if (experimentOn) return          // 实验模式的队列由清单决定
    const timer = setTimeout(() => void loadQueue(), 250)
    return () => clearTimeout(timer)
  }, [queueQuery, experimentOn, loadQueue])

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

      {/* 文本搜索：只在普通标注模式出现（实验模式的队列由清单决定） */}
      {!experimentOn && (
        <div className="queue-search">
          <input
            type="search"
            value={queueQuery}
            placeholder="搜索图片名 / id…"
            aria-label="搜索图片"
            onChange={(e) => setQueueQuery(e.target.value)}
          />
          {queueQuery !== '' && (
            <button
              type="button"
              className="queue-search-clear"
              onClick={() => setQueueQuery('')}
              title="清空搜索"
              aria-label="清空搜索"
            >
              ×
            </button>
          )}
        </div>
      )}

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
          onChange={(e) => setStrategy(e.target.value as StrategyChoice)}
          disabled={queueLoading}
          title="主动学习选样策略"
        >
          {(Object.keys(STRATEGY_LABEL) as StrategyChoice[]).map((s) => (
            <option key={s} value={s}>
              {STRATEGY_LABEL[s]}
            </option>
          ))}
        </select>
        <input
          type="number"
          className="sample-k"
          min={1}
          max={200}
          value={sampleK}
          disabled={queueLoading || strategy === 'all'}
          onChange={(e) => setSampleK(clampK(Number(e.target.value)))}
          title="取多少张。设得比池子大就取整池（按难度排好序）——三臂实验请用左下角的「实验模式」，不要用智能选样"
        />
        <button
          onClick={() =>
            strategy === 'all'
              ? void loadQueue()
              : void loadSample(strategy, sampleK)
          }
          disabled={queueLoading}
          title={
            strategy === 'all'
              ? '按路径顺序恢复整个队列（清掉选样结果与分数）'
              : `调 /api/sample，把这批里最值得标的 ${sampleK} 张换进队列`
          }
        >
          {strategy === 'all' ? '显示全部' : '智能选样'}
        </button>
      </div>

      <div className="queue">
        {queueLoading && <div className="queue-empty">加载中…</div>}
        {!queueLoading && queue.length === 0 && (
          <div className="queue-empty">队列为空，检查后端 /api/images</div>
        )}
        {queue.map((it, i) => {
          const active = i === index
          // 同图三臂：同一张图在队列里出现 3 次（三个臂各一次），
          // 所以 key 和"是否完成"都必须带上臂，否则三个任务会被当成同一个。
          const taskArm = experimentOn ? expTasks[i]?.arm : undefined
          const done = experimentOn
            ? !!expDone[taskKey(it.image_id, taskArm ?? '')]
            : it.image_id in submitted
          return (
            <button
              key={`${it.image_id}#${i}`}
              className={`queue-item${active ? ' on' : ''}${done ? ' done' : ''}`}
              onClick={() => void openIndex(i)}
              title={`${it.image_id} · ${it.width}×${it.height}` +
                (taskArm ? ` · ${ARM_LABEL[taskArm]}` : '')}
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
              {taskArm && (
                <span className={`queue-arm arm-${taskArm}`} title={ARM_LABEL[taskArm]}>
                  {ARM_SHORT[taskArm]}
                </span>
              )}
              <span className="queue-cls">{clsName.get(it.gt_class_id ?? -1) ?? '—'}</span>
              {done && <span className="queue-tick">✓</span>}
            </button>
          )
        })}
      </div>

      {/* 左下角：明暗切换 + 实验模式 + 加载更多 + 队列说明 */}
      <div className="pane-foot queue-foot">
        <ThemeToggle />
        <ExperimentPanel />
        {!experimentOn && queueHasMore && (
          <button
            type="button"
            className="queue-more"
            onClick={() => void loadMoreQueue()}
            disabled={queueLoading}
            title="再载入 60 张（滚到列表底部也会自动加载）"
          >
            加载更多
          </button>
        )}
        <span className="foot-note">
          {experimentOn
            ? `实验任务 ${queue.length} 个`
            : `已载入 ${queue.length} / ${queueTotal} 张`}
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

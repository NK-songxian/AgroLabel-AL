import { useEffect, useMemo, useRef, useState } from 'react'

import { useStore } from '../store'
import { ARM_LABEL, taskKey, type Arm } from '../types'

/**
 * 实验模式入口。位置在左栏底部、明暗切换的右边。
 *
 * 它把"三臂对照实验"需要的东西收在一处，因为散着放一定会标错：
 *   - 读实验任务清单（同图三臂：一张图 3 个任务，三个臂各标一遍）
 *   - 队列换成清单里的任务（否则要在 7773 张里自己找）
 *   - 每臂 / 每组进度
 *   - 实验臂选择器（**始终可见**；实验模式下锁定，因为臂由任务决定）
 *
 * 没载入清单时也能点开：里面会说清单该放哪、怎么用环境变量指路。
 */
export function ExperimentPanel() {
  const experiment = useStore((s) => s.experiment)
  const experimentOn = useStore((s) => s.experimentOn)
  const expTasks = useStore((s) => s.expTasks)
  const expDone = useStore((s) => s.expDone)
  const index = useStore((s) => s.index)
  const arm = useStore((s) => s.arm)
  const setArm = useStore((s) => s.setArm)
  const loadExperiment = useStore((s) => s.loadExperiment)
  const exitExperiment = useStore((s) => s.exitExperiment)

  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement | null>(null)

  // 点面板外面就收起。用 mousedown 而不是 click：
  // 用 click 的话，点按钮时"外面的 click"会先把面板关上，看起来像没响应。
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const loaded = experiment?.loaded === true
  const totalTasks = loaded ? experiment.n_tasks : 0
  const doneN = Object.keys(expDone).length

  // 每臂进度：分母来自清单里的任务数，分子来自本地已完成集合
  const perArm = useMemo(() => {
    const out: Record<string, { total: number; done: number }> = {}
    for (const t of experiment?.tasks ?? []) {
      if (!out[t.arm]) out[t.arm] = { total: 0, done: 0 }
      out[t.arm].total += 1
      if (expDone[taskKey(t.image_id, t.arm)]) out[t.arm].done += 1
    }
    return out
  }, [experiment, expDone])

  // 当前任务（同图三臂时能看到"这是哪张图的哪个臂"）
  const curTask = experimentOn ? expTasks[index] : undefined

  return (
    <div className="exp-wrap" ref={wrapRef}>
      <button
        className={`exp-toggle${experimentOn ? ' on' : ''}`}
        onClick={() => setOpen((v) => !v)}
        title={
          experimentOn
            ? `实验模式已开启：${doneN}/${totalTasks} 个任务完成`
            : '实验模式：载入三臂对照的任务清单，按任务自动切换实验臂'
        }
        aria-expanded={open}
      >
        <span className="glyph" aria-hidden="true">
          {experimentOn ? '●' : '◇'}
        </span>
        {experimentOn ? `实验 ${doneN}/${totalTasks}` : '实验模式'}
      </button>

      {open && (
        <div className="exp-pop" role="dialog" aria-label="实验模式">
          <div className="exp-head">
            三臂对照实验
            {loaded && <span className="exp-badge">{experiment.name ?? '已载入'}</span>}
          </div>

          {/*
            臂选择只在**实验模式**出现。
            普通标注模式不暴露"臂"这个概念 —— 进来就是预标注 + 点提示全可用，
            不需要先去面板里挑一下。
          */}
          {experimentOn && (
            <label className="exp-arm">
              <span>切到臂</span>
              <select
                value={arm}
                onChange={(e) => setArm(e.target.value as Arm)}
                title="选臂 = 跳到这个臂的下一个未完成任务（臂始终跟着任务走，不会把产物落到错目录）"
              >
                {(Object.keys(ARM_LABEL) as Arm[]).map((a) => (
                  <option key={a} value={a}>
                    {ARM_LABEL[a]}
                  </option>
                ))}
              </select>
            </label>
          )}

          {!experimentOn && (
            <p className="exp-note">
              当前是<strong>普通标注模式</strong>：预标注与点提示都可用，
              标注结果落到 <code>out/annotations/assisted/</code>。
              点下面的按钮才进入三臂对照实验。
            </p>
          )}

          {!loaded && (
            <>
              <p className="exp-note">
                {experiment?.note || '还没有载入实验任务清单。'}
              </p>
              <p className="exp-note dim">
                做法：把清单放到 <code>out/experiment/manifest.json</code>（或用环境变量
                <code> AGROLABEL_EXPERIMENT</code> 指到别处），然后点下面这个按钮。
              </p>
              <div className="exp-actions">
                <button className="btn primary" onClick={() => void loadExperiment()}>
                  读清单
                </button>
              </div>
            </>
          )}

          {loaded && (
            <>
              <div className="exp-sub">
                {experiment.mode === 'tasks' ? '同图三臂' : '一图一臂'} ·{' '}
                {experiment.n_images} 张图 / {totalTasks} 个任务 · 已完成 {doneN}
                <span className="exp-src" title={experiment.source ?? ''}>
                  {experiment.source}
                </span>
              </div>

              <div className="exp-arms">
                {(Object.keys(ARM_LABEL) as Arm[])
                  .filter((a) => perArm[a])
                  .map((a) => {
                    const p = perArm[a]
                    const pct = p.total ? Math.round((p.done / p.total) * 100) : 0
                    return (
                      <div className="exp-row" key={a}>
                        <span className="exp-row-label">{ARM_LABEL[a]}</span>
                        <span className="exp-bar">
                          <i style={{ width: `${pct}%` }} />
                        </span>
                        <span className="exp-row-num">
                          {p.done}/{p.total}
                        </span>
                      </div>
                    )
                  })}
              </div>

              {Object.keys(experiment.groups).length > 1 && (
                <div className="exp-groups">
                  {Object.entries(experiment.groups).map(([g, v]) => (
                    <span key={g} className="exp-group">
                      {g} {v.done}/{v.n}
                    </span>
                  ))}
                </div>
              )}

              {curTask && (
                <p className="exp-note dim">
                  当前任务：{curTask.group ?? '未分组'} · 第 {index + 1}/{totalTasks} 个 ·{' '}
                  {ARM_LABEL[curTask.arm]}
                </p>
              )}

              {experimentOn && (
                <p className="exp-note dim">
                  臂跟着任务走；选臂 = 跳到该臂的下一个任务。
                  同一张图会在三个臂各出现一次，这是设计（同图对照）。
                </p>
              )}

              {experiment.problems.length > 0 && (
                <p className="exp-note err" title={experiment.problems.join('\n')}>
                  清单有 {experiment.problems.length} 条被跳过（悬停看详情）
                </p>
              )}

              <div className="exp-actions">
                {experimentOn ? (
                  <>
                    <button className="btn" onClick={() => void loadExperiment()}>
                      重新载入
                    </button>
                    <button className="btn" onClick={exitExperiment}>
                      退出实验模式
                    </button>
                  </>
                ) : (
                  <button className="btn primary" onClick={() => void loadExperiment()}>
                    开启实验模式
                  </button>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

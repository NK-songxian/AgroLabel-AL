import { useEffect, useMemo, useRef, useState } from 'react'

import { useStore } from '../store'
import { ARM_LABEL, type Arm } from '../types'

/**
 * 实验模式入口。位置在左栏底部、明暗切换的右边。
 *
 * 它把"三臂对照实验"需要的东西收在一处，因为散着放一定会标错：
 *   - 读实验清单（68 张 val 图，每张绑定一个臂）
 *   - 队列换成清单里的图（否则要在 7773 张里自己找）
 *   - 每臂进度（这臂还剩几张）
 *   - 实验臂选择器（搬到这里；实验模式下锁定，由图清单决定）
 *
 * 没载入清单时也能点开：里面会说清楚清单该放哪、怎么用环境变量指路。
 */
export function ExperimentPanel() {
  const experiment = useStore((s) => s.experiment)
  const experimentOn = useStore((s) => s.experimentOn)
  const expDone = useStore((s) => s.expDone)
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
  const doneN = Object.keys(expDone).length
  const totalN = loaded ? experiment.n_images : 0

  const perArm = useMemo(() => {
    const out: Record<string, { total: number; done: number }> = {}
    for (const im of experiment?.images ?? []) {
      if (!out[im.arm]) out[im.arm] = { total: 0, done: 0 }
      out[im.arm].total += 1
      if (expDone[im.image_id]) out[im.arm].done += 1
    }
    return out
  }, [experiment, expDone])

  return (
    <div className="exp-wrap" ref={wrapRef}>
      <button
        className={`exp-toggle${experimentOn ? ' on' : ''}`}
        onClick={() => setOpen((v) => !v)}
        title={
          experimentOn
            ? `实验模式已开启：${doneN}/${totalN} 张已完成`
            : '实验模式：载入三臂对照的图清单，按图自动绑定实验臂'
        }
        aria-expanded={open}
      >
        <span className="glyph" aria-hidden="true">
          {experimentOn ? '●' : '◇'}
        </span>
        {experimentOn ? `实验 ${doneN}/${totalN}` : '实验模式'}
      </button>

      {open && (
        <div className="exp-pop" role="dialog" aria-label="实验模式">
          <div className="exp-head">
            三臂对照实验
            {loaded && <span className="exp-badge">{experiment.name ?? '已载入'}</span>}
          </div>

          {!loaded && (
            <>
              <p className="exp-note">
                {experiment?.note || '还没有载入实验清单。'}
              </p>
              <p className="exp-note dim">
                做法：把实验包的 <code>manifest.json</code> 放到
                <code> out/experiment/manifest.json</code>（或用环境变量
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
                清单 {totalN} 张 · 已完成 {doneN} / {totalN}
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

              <label className="exp-arm">
                <span>实验臂</span>
                <select
                  value={arm}
                  disabled={experimentOn}
                  onChange={(e) => setArm(e.target.value as Arm)}
                  title={
                    experimentOn
                      ? '实验模式下臂由图清单决定，不能手动切换'
                      : '切换后当前图按新条件重新载入，产物分别落到 out/annotations/<arm>/'
                  }
                >
                  {(Object.keys(ARM_LABEL) as Arm[]).map((a) => (
                    <option key={a} value={a}>
                      {ARM_LABEL[a]}
                    </option>
                  ))}
                </select>
              </label>
              {experimentOn && (
                <p className="exp-note dim">
                  当前图所属的臂已自动切换并锁定；退出实验模式后可手动改。
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

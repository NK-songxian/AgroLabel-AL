import { Select } from 'antd'
import { useMemo, useState } from 'react'

import { classSolid } from '../colors'
import { useStore } from '../store'

function download(name: string, text: string) {
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function InspectorPanel() {
  const current = useStore((s) => s.current)
  const instances = useStore((s) => s.instances)
  const selectedId = useStore((s) => s.selectedId)
  const select = useStore((s) => s.select)
  const changeClass = useStore((s) => s.changeClass)
  const deleteSelected = useStore((s) => s.deleteSelected)
  const confirmAll = useStore((s) => s.confirmAll)
  const markNoTarget = useStore((s) => s.markNoTarget)
  const undo = useStore((s) => s.undo)
  const submit = useStore((s) => s.submit)
  const classes = useStore((s) => s.classes)
  const labelState = useStore((s) => s.labelState)
  const nOps = useStore((s) => s.nOps)
  const latencyMs = useStore((s) => s.latencyMs)
  const modelVersion = useStore((s) => s.modelVersion)
  const status = useStore((s) => s.status)
  const ops = useStore((s) => s.ops)
  const countedForCurrent = useStore((s) => s.countedForCurrent)
  const exportOpsCsv = useStore((s) => s.exportOpsCsv)
  const toast = useStore((s) => s.toast)

  const [copied, setCopied] = useState(false)

  const options = useMemo(
    () => classes.map((c) => ({ value: c.class_id, label: c.class_name })),
    [classes],
  )

  const busy = status === 'predicting'
  const counted = countedForCurrent()

  if (!current) {
    return (
      <aside className="pane pane-right">
        <div className="pane-head">
          <span>详情</span>
        </div>
        <div className="pane-empty">未选择图像</div>
      </aside>
    )
  }

  return (
    <aside className="pane pane-right">
      <div className="pane-head">
        <span>详情</span>
        <span className="pane-count">{instances.length} 实例</span>
      </div>

      <div className="imgmeta">
        <div className="imgmeta-name" title={current.file_name}>
          {current.file_name}
        </div>
        <div className="imgmeta-row">
          <span>{current.width} × {current.height}</span>
          <span className="sep">·</span>
          <span>{current.split}</span>
          {current.disease && (
            <>
              <span className="sep">·</span>
              <span title="来自 COCO 清单，不属于 34 类">{current.disease}</span>
            </>
          )}
        </div>
        <div className="imgmeta-row dim">
          {latencyMs != null && <span>预标注 {latencyMs.toFixed(1)} ms</span>}
          {modelVersion && (
            <span className="mono" title={modelVersion}>
              {modelVersion.slice(0, 26)}…
            </span>
          )}
        </div>
      </div>

      {labelState === 'no_target' && (
        <div className="banner warn">
          已标记为「无目标」——这张图不会计入漏检，但会计入分母
        </div>
      )}

      <div className="pane-subhead">
        <span>实例</span>
        {selectedId && (
          <button className="link danger" onClick={deleteSelected} title="删除 (Del)">
            删除选中
          </button>
        )}
      </div>

      <div className="instances">
        {instances.length === 0 && (
          <div className="pane-empty">
            {busy ? '预标注中…' : '没有实例。模型没检出，或你已删空。'}
          </div>
        )}
        {instances.map((inst, i) => {
          const sel = inst.id === selectedId
          return (
            <div
              key={inst.id}
              className={`inst${sel ? ' on' : ''}`}
              onClick={() => select(inst.id)}
            >
              <span className="inst-idx" style={{ background: classSolid(inst.class_id) }}>
                {i + 1}
              </span>
              <Select
                size="small"
                className="inst-select"
                value={inst.class_id}
                options={options}
                showSearch
                optionFilterProp="label"
                onChange={(v) => changeClass(inst.id, v)}
                popupMatchSelectWidth={200}
              />
              <span className={`inst-score${inst.source === 'manual' ? ' manual' : ''}`}>
                {inst.score != null ? inst.score.toFixed(2) : '人'}
              </span>
            </div>
          )
        })}
      </div>

      <div className="actions">
        <button className="btn" onClick={confirmAll} disabled={busy} title="Enter">
          整张确认
        </button>
        <button className="btn" onClick={markNoTarget} disabled={busy} title="这张图没有目标">
          标为无目标
        </button>
        <button className="btn" onClick={undo} disabled={busy} title="Ctrl+Z">
          撤销
        </button>
        <button className="btn primary" onClick={() => void submit()} disabled={busy}>
          提交并下一张
        </button>
      </div>

      <div className="stats">
        <div className="stat">
          <span className="stat-num">{counted}</span>
          <span className="stat-label">计入修改次数</span>
        </div>
        <div className="stat">
          <span className="stat-num">{nOps}</span>
          <span className="stat-label">本轮操作</span>
        </div>
        <div className="stat">
          <span className="stat-num">{ops.length}</span>
          <span className="stat-label">会话日志条数</span>
        </div>
      </div>

      <div className="pane-foot ops-foot">
        <button
          className="link"
          disabled={!ops.length}
          onClick={() => {
            download(`ops_${Date.now()}.csv`, exportOpsCsv())
            toast('已下载 ops.csv', 'ok')
          }}
        >
          下载 ops.csv
        </button>
        <button
          className="link"
          disabled={!ops.length}
          onClick={() => {
            void navigator.clipboard.writeText(exportOpsCsv()).then(() => {
              setCopied(true)
              setTimeout(() => setCopied(false), 1500)
            })
          }}
        >
          {copied ? '已复制' : '复制 CSV'}
        </button>
      </div>
      <div className="pane-foot dim tiny">
        表头与 src/m4_app.py:131 一致，可直接追加到 out/logs/ops.csv
      </div>
    </aside>
  )
}

import { Select } from 'antd'
import { useMemo, useState } from 'react'

import { api } from '../api'
import { classSolid } from '../colors'
import { useStore } from '../store'
import { ARM_LABEL } from '../types'
import type { Arm } from '../types'

function download(name: string, text: string) {
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/**
 * 从后端导出该臂的完整 ops.csv（跨会话）。
 *
 * 后端的 `/api/logs/export` 直接回原始字节流（保留 UTF-8 BOM，Excel 打开不乱码），
 * 比前端内存里那几条更完整 —— 一次实验会跨多次刷新/多个会话。
 * 后端不可用时退回本会话的日志，不让这个按钮变成"点了没反应"。
 */
async function downloadServerOps(arm: string) {
  const toast = useStore.getState().toast
  try {
    const res = await fetch(api.logsExportUrl('csv', arm))
    if (!res.ok) throw new Error(String(res.status))
    const blob = await res.blob()
    // 文件名从 Content-Disposition 取，后端带了时间戳
    const cd = res.headers.get('Content-Disposition') ?? ''
    const m = /filename="?([^"]+)"?/.exec(cd)
    const name = m ? m[1] : `ops_${arm}_${Date.now()}.csv`

    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = name
    a.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)

    const rows = res.headers.get('X-Row-Count')
    toast(`已从服务端导出 ${name}（${rows ?? '?'} 条）`, 'ok')
  } catch {
    download(`ops_${Date.now()}.csv`, useStore.getState().exportOpsCsv())
    toast('后端导出不可用，已改为下载本会话的日志', 'warn')
  }
}

/** 实验臂切换条。三臂对照的实验条件，切了会重新载入当前图。 */
function ArmBar() {
  const arm = useStore((s) => s.arm)
  const setArm = useStore((s) => s.setArm)
  return (
    <div className="arm-bar">
      <span className="arm-label">实验臂</span>
      <select
        value={arm}
        onChange={(e) => setArm(e.target.value as Arm)}
        title="三臂对照用：切换后当前图按新条件重新载入，产物分别落到 out/annotations/<arm>/"
      >
        {(Object.keys(ARM_LABEL) as Arm[]).map((a) => (
          <option key={a} value={a}>
            {ARM_LABEL[a]}
          </option>
        ))}
      </select>
    </div>
  )
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
  const arm = useStore((s) => s.arm)
  const countedForCurrent = useStore((s) => s.countedForCurrent)
  const exportOpsCsv = useStore((s) => s.exportOpsCsv)

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
      <ArmBar />

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
          onClick={() => void downloadServerOps(arm)}
          title="从后端导出该臂的完整日志（跨会话），拿不到时退回本会话的日志"
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

      {/* 画布上的鼠标操作速查。每个模式都列出来，省得来回切着试 */}
      <div className="mouse-legend">
        <span>
          <kbd>左键</kbd>选中 / 拖顶点
        </span>
        <span>
          <kbd>右键双击</kbd>加节点
        </span>
        <span>
          <kbd>中键拖拽</kbd>平移
        </span>
        <span>
          <kbd>滚轮</kbd>缩放
        </span>
        <span>
          <kbd>左键双击</kbd>自适应
        </span>
      </div>

      <div className="pane-foot dim tiny">
        表头与 src/m4_app.py:131 一致，可直接追加到 out/logs/ops.csv
      </div>
    </aside>
  )
}

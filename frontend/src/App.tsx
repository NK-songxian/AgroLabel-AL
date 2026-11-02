import { useEffect } from 'react'

import { CanvasStage } from './components/CanvasStage'
import { InspectorPanel } from './components/InspectorPanel'
import { QueuePanel } from './components/QueuePanel'
import { TopBar } from './components/TopBar'
import { useHotkeys } from './hooks/useHotkeys'
import { useStore } from './store'

function Toasts() {
  const toasts = useStore((s) => s.toasts)
  const dismiss = useStore((s) => s.dismissToast)
  if (!toasts.length) return null
  return (
    <div className="toasts">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} onClick={() => dismiss(t.id)}>
          {t.text}
        </div>
      ))}
    </div>
  )
}

function BootScreen() {
  const bootError = useStore((s) => s.bootError)
  const status = useStore((s) => s.status)
  const bootstrap = useStore((s) => s.bootstrap)

  return (
    <div className="boot">
      <div className="boot-card">
        <div className="boot-title">AgroLabel-AL</div>
        {bootError ? (
          <>
            <div className="boot-err">连不上后端</div>
            <pre className="boot-detail">{bootError}</pre>
            <div className="boot-hint">
              先在仓库根目录启动：
              <code>.\.venv\Scripts\python.exe -m uvicorn backend.app:app --port 8000</code>
            </div>
            <button className="btn primary" onClick={() => void bootstrap()}>
              重试
            </button>
          </>
        ) : (
          <>
            <div className="spinner" />
            <div className="boot-hint">{status === 'booting' ? '连接后端…' : '加载中…'}</div>
          </>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const bootstrap = useStore((s) => s.bootstrap)
  const health = useStore((s) => s.health)
  const ready = health !== null

  useHotkeys()

  useEffect(() => {
    void bootstrap()
  }, [bootstrap])

  return (
    <div className="app">
      {!ready && <BootScreen />}
      {ready && (
        <>
          <TopBar />
          <div className="body">
            <QueuePanel />
            <main className="pane pane-center">
              <CanvasStage />
            </main>
            <InspectorPanel />
          </div>
        </>
      )}
      <Toasts />
    </div>
  )
}

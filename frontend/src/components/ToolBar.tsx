import { useStore } from '../store'

/**
 * 顶栏工具组：画布交互模式 + 视图开关 + 顶点编辑。
 *
 * 从画布左上角的浮层挪到顶栏，是为了不遮住图像 ——
 * 病斑面积中位数只有整图的 0.3%，画面左上角正是常有病灶的位置。
 */
export function ToolBar() {
  const current = useStore((s) => s.current)
  const mode = useStore((s) => s.mode)
  const refining = useStore((s) => s.refining)
  const showMasks = useStore((s) => s.showMasks)
  const draft = useStore((s) => s.draft)
  const selectedId = useStore((s) => s.selectedId)
  const arm = useStore((s) => s.arm)
  const operator = useStore((s) => s.operator)
  const setOperator = useStore((s) => s.setOperator)
  const setMode = useStore((s) => s.setMode)
  const toggleMasks = useStore((s) => s.toggleMasks)
  const addVertex = useStore((s) => s.addVertex)
  const toast = useStore((s) => s.toast)

  if (!current) return <div className="topbar-tools" />

  const canAddVertex = showMasks && selectedId != null

  return (
    <div className="topbar-tools">
      {/* 标注员：写进每张图的元数据。三臂交叉分配时要按人统计，
          所以必须在「选择」左边、一眼能看到的地方 */}
      <input
        className="operator-input"
        value={operator}
        placeholder="标注员"
        maxLength={32}
        aria-label="标注员"
        title="标注员姓名或编号。会写进每张图的元数据（out/annotations/<arm>/<id>.json），交叉分配时按人统计"
        onChange={(e) => setOperator(e.target.value)}
      />

      <span className="tool-sep" />

      <div className="tool-group">
        <button
          className={`tool${mode === 'select' ? ' on' : ''}`}
          onClick={() => setMode('select')}
          title="点选 / 拖顶点（V）"
        >
          选择
        </button>
        <button
          className={`tool${mode === 'refine' ? ' on' : ''}`}
          disabled={arm === 'noprompt'}
          onClick={() => setMode('refine')}
          title={
            arm === 'noprompt'
              ? '当前是「预标 · 禁用点提示」臂，点提示不可用（切到别的臂才能用）'
              : '点一下病斑，让模型在放大后的局部重新识别（P）'
          }
        >
          点提示
        </button>
        <button
          className={`tool${mode === 'draw' ? ' on' : ''}`}
          onClick={() => setMode('draw')}
          title="手画多边形：逐点单击，右键双击或点回起点闭合（D）"
        >
          手画
        </button>
      </div>

      <span className="tool-sep" />

      <div className="tool-group">
        <button
          className={`tool toggle${showMasks ? ' on' : ''}`}
          onClick={toggleMasks}
          title="显示 / 隐藏标注框（H）—— 关掉可以干净地看原图，不影响已标的实例"
          aria-pressed={showMasks}
        >
          <span className="tool-eye" aria-hidden>
            {showMasks ? '◉' : '○'}
          </span>
          标注{showMasks ? '显示' : '隐藏'}
        </button>
        <button
          className="tool"
          disabled={!canAddVertex}
          onClick={() => {
            if (!addVertex()) toast('先选中一个实例，再加节点', 'warn')
          }}
          title={
            canAddVertex
              ? '在选中多边形最长的边上插一个顶点，然后拖到准确位置'
              : '先选中一个实例才能加节点'
          }
        >
          ＋ 加节点
        </button>
        <button
          className={`tool${mode === 'erase' ? ' on' : ''}`}
          disabled={!canAddVertex && mode !== 'erase'}
          onClick={() => setMode(mode === 'erase' ? 'select' : 'erase')}
          title={
            canAddVertex || mode === 'erase'
              ? '橡皮擦：点节点把它擦掉（X）。少于 3 个顶点时不允许'
              : '先选中一个实例才能删节点'
          }
          aria-pressed={mode === 'erase'}
        >
          ⌫ 删节点
        </button>
      </div>

      {mode === 'select' && (
        <span className="tool-hint">
          左键选中/拖顶点 · 右键双击加节点 · 中键平移
        </span>
      )}
      {mode === 'refine' && (
        <span className="tool-hint">
          左键点一下病斑{refining && ' · 识别中…'}
        </span>
      )}
      {mode === 'draw' && (
        <span className="tool-hint">
          {draft && draft.length > 0
            ? `${draft.length} 个点 · 右键双击闭合 · Backspace 退一点 · Esc 取消`
            : '左键逐点单击，≥3 点后右键双击闭合'}
        </span>
      )}
      {mode === 'erase' && (
        <span className="tool-hint">
          左键点节点擦除 · 只剩 3 个顶点时停手 · Esc 退出
        </span>
      )}
    </div>
  )
}

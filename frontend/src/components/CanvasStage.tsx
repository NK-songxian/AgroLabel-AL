/**
 * Konva 画布：显示原图 + 叠加预标注多边形。
 *
 * 交互（当前阶段）：
 *   - 滚轮缩放（以光标为锚点）
 *   - 拖拽平移
 *   - 点多边形选中、点空白取消
 *   - 双击 / F 键 自适应
 *
 * 坐标约定：后端给的是**归一化**多边形，这里统一乘原图宽高还原成像素，
 * 再交给 Group 的 scale 变换。不要在这里做任何归一化又归一化的来回转换。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Circle, Group, Image as KonvaImage, Layer, Line, Stage, Text } from 'react-konva'

import type Konva from 'konva'

import { api } from '../api'
import { classFill, classStroke } from '../colors'
import { useStore } from '../store'
import type { Instance } from '../types'

const MIN_SCALE = 0.05
const MAX_SCALE = 24
/** 自适应时最多放大到 4x —— 数据集里有 87px 的图，无上限会糊成一团 */
const FIT_MAX_SCALE = 4
const FIT_PADDING = 28

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

export function CanvasStage() {
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const stageRef = useRef<Konva.Stage | null>(null)

  const current = useStore((s) => s.current)
  const instances = useStore((s) => s.instances)
  const selectedId = useStore((s) => s.selectedId)
  const status = useStore((s) => s.status)
  const scale = useStore((s) => s.scale)
  const tx = useStore((s) => s.tx)
  const ty = useStore((s) => s.ty)
  const fitToken = useStore((s) => s.fitToken)
  const select = useStore((s) => s.select)
  const setViewport = useStore((s) => s.setViewport)
  const mode = useStore((s) => s.mode)
  const refining = useStore((s) => s.refining)
  const refineAt = useStore((s) => s.refineAt)
  const showMasks = useStore((s) => s.showMasks)
  const toggleMasks = useStore((s) => s.toggleMasks)
  const draft = useStore((s) => s.draft)
  const drawAddPoint = useStore((s) => s.drawAddPoint)
  const drawCommit = useStore((s) => s.drawCommit)
  const beginVertexDrag = useStore((s) => s.beginVertexDrag)
  const moveVertex = useStore((s) => s.moveVertex)
  const endVertexDrag = useStore((s) => s.endVertexDrag)

  const [size, setSize] = useState({ width: 0, height: 0 })
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  const [imgState, setImgState] = useState<'idle' | 'loading' | 'ok' | 'err'>('idle')

  // ---- 容器尺寸
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      setSize({ width: el.clientWidth, height: el.clientHeight })
    })
    ro.observe(el)
    setSize({ width: el.clientWidth, height: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  // ---- 图片加载
  // 这里的 setState 是"与外部系统同步"（HTMLImageElement 的异步加载），
  // 不是能挪到事件里去的派生状态，所以针对这条规则做了豁免。
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (!current) {
      setImg(null)
      setImgState('idle')
      return
    }
    let alive = true
    setImgState('loading')
    const el = new window.Image()
    el.decoding = 'sync'
    el.onload = () => {
      if (!alive) return
      setImg(el)
      setImgState('ok')
    }
    el.onerror = () => {
      if (!alive) return
      setImg(null)
      setImgState('err')
    }
    el.src = api.imageUrl(current)
    return () => {
      alive = false
      el.onload = null
      el.onerror = null
    }
  }, [current])
  /* eslint-enable react-hooks/set-state-in-effect */

  const imgW = current?.width || img?.naturalWidth || 0
  const imgH = current?.height || img?.naturalHeight || 0

  // ---- 自适应
  const fit = useCallback(() => {
    if (!imgW || !imgH || !size.width || !size.height) return
    const s = clamp(
      Math.min(
        (size.width - FIT_PADDING * 2) / imgW,
        (size.height - FIT_PADDING * 2) / imgH,
      ),
      MIN_SCALE,
      FIT_MAX_SCALE,
    )
    setViewport({
      scale: s,
      tx: (size.width - imgW * s) / 2,
      ty: (size.height - imgH * s) / 2,
    })
  }, [imgW, imgH, size.width, size.height, setViewport])

  useEffect(() => {
    fit()
  }, [fit, fitToken, imgState])

  // ---- 滚轮缩放
  const onWheel = useCallback(
    (e: Konva.KonvaEventObject<WheelEvent>) => {
      e.evt.preventDefault()
      const stage = stageRef.current
      const p = stage?.getPointerPosition()
      if (!p) return

      const factor = e.evt.deltaY > 0 ? 1 / 1.14 : 1.14
      const nextScale = clamp(scale * factor, MIN_SCALE, MAX_SCALE)
      if (nextScale === scale) return

      // 让光标下的像素点在缩放前后保持不动
      const mx = (p.x - tx) / scale
      const my = (p.y - ty) / scale
      setViewport({
        scale: nextScale,
        tx: p.x - mx * nextScale,
        ty: p.y - my * nextScale,
      })
    },
    [scale, tx, ty, setViewport],
  )

  // 归一化 -> 像素
  const pixelOf = useCallback(
    (inst: Instance) => inst.polygon.flatMap(([x, y]) => [x * imgW, y * imgH]),
    [imgW, imgH],
  )

  const selected = useMemo(
    () => instances.find((x) => x.id === selectedId) ?? null,
    [instances, selectedId],
  )

  const strokeUnit = 1 / Math.max(scale, 0.001)

  /** 屏幕坐标 -> 图像像素坐标；出图返回 null */
  const toImagePx = useCallback(
    (p: { x: number; y: number }): { x: number; y: number } | null => {
      const ix = (p.x - tx) / scale
      const iy = (p.y - ty) / scale
      if (ix < 0 || iy < 0 || ix > imgW || iy > imgH) return null
      return { x: ix, y: iy }
    },
    [tx, ty, scale, imgW, imgH],
  )

  /**
   * 点空白处才会走到这里 —— 实例的 Line 已经 cancelBubble 了。
   * refine 模式：反算成图像像素坐标交给点提示。
   * draw 模式：落一个顶点；点回起点则闭合。
   */
  const onStageClick = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      const stage = stageRef.current
      const p = stage?.getPointerPosition()
      if (!p) return

      if (mode === 'draw') {
        if (!imgW || !imgH) return
        const px = toImagePx(p)
        if (!px) return
        const d = draft ?? []

        // 点回第一个点 = 闭合（屏幕距离 10px 内算命中）
        if (d.length >= 3) {
          const [fx, fy] = d[0]
          if (Math.hypot(px.x - fx * imgW, px.y - fy * imgH) * scale < 10) {
            drawCommit()
            return
          }
        }
        // 与上一点几乎重合（双击的第二下）就忽略，避免插一个零长边
        const last = d[d.length - 1]
        if (last) {
          const [lx, ly] = last
          if (Math.hypot(px.x - lx * imgW, px.y - ly * imgH) * scale < 4) return
        }
        drawAddPoint(px.x / imgW, px.y / imgH)
        return
      }

      if (mode !== 'refine') {
        if (e.target === e.target.getStage()) select(null)
        return
      }
      if (refining || !imgW || !imgH) return

      const px = toImagePx(p)
      if (!px) return
      void refineAt(px.x, px.y)
    },
    [mode, refining, imgW, imgH, draft, scale, select, refineAt,
     drawAddPoint, drawCommit, toImagePx],
  )

  return (
    <div className="stage-wrap" ref={wrapRef}>
      {size.width > 0 && (
        <Stage
          ref={stageRef}
          width={size.width}
          height={size.height}
          onWheel={onWheel}
          onClick={onStageClick}
          onDblClick={() => {
            // draw 模式下双击 = 闭合，不能顺手把视图重置了
            if (mode === 'draw') {
              if ((draft?.length ?? 0) >= 3) drawCommit()
              else useStore.getState().drawCancel()
              return
            }
            fit()
          }}
          style={{
            cursor: mode === 'draw' ? 'crosshair' : mode === 'refine' ? 'copy' : 'default',
          }}
        >
          <Layer>
            <Group
              x={tx}
              y={ty}
              scaleX={scale}
              scaleY={scale}
              draggable
              onDragEnd={(e) =>
                setViewport({ tx: e.target.x(), ty: e.target.y() })
              }
            >
              {img && (
                <KonvaImage image={img} width={imgW} height={imgH} listening={false} />
              )}

              {/* 关掉标注时连实例一起不渲染：既不挡视线，也不再拦截点击
                  （于是 select 模式的点空白取消、refine 模式的点提示都还能用） */}
              {showMasks && instances.map((inst, i) => {
                const isSel = inst.id === selectedId
                return (
                  <Group key={inst.id}>
                    <Line
                      points={pixelOf(inst)}
                      closed
                      fill={classFill(inst.class_id, isSel ? 0.34 : 0.2)}
                      stroke={classStroke(inst.class_id, isSel ? 1 : 0.85)}
                      strokeWidth={(isSel ? 2.5 : 1.4) * strokeUnit}
                      hitStrokeWidth={14 * strokeUnit}
                      perfectDrawEnabled={false}
                      onClick={(e) => {
                        e.cancelBubble = true
                        select(inst.id)
                      }}
                      onTap={(e) => {
                        e.cancelBubble = true
                        select(inst.id)
                      }}
                    />
                    {/* 序号：方便对日志里的 instance 字段 */}
                    <Text
                      text={String(i + 1)}
                      x={inst.polygon.reduce((a, p) => a + p[0], 0) / inst.polygon.length * imgW}
                      y={inst.polygon.reduce((a, p) => a + p[1], 0) / inst.polygon.length * imgH}
                      fontSize={11 * strokeUnit}
                      fill={classStroke(inst.class_id)}
                      listening={false}
                      offsetX={3 * strokeUnit}
                      offsetY={5 * strokeUnit}
                    />
                  </Group>
                )
              })}

              {/* 选中实例的顶点：可拖动微调边界。整段拖动只在 dragStart 压一次撤销栈 */}
              {showMasks && mode === 'select' && selected?.polygon.map(([x, y], i) => (
                <Circle
                  key={`${selected.id}-${i}`}
                  x={x * imgW}
                  y={y * imgH}
                  radius={4 * strokeUnit}
                  fill="#fff"
                  stroke={classStroke(selected.class_id)}
                  strokeWidth={1.4 * strokeUnit}
                  draggable
                  onMouseEnter={(e) => {
                    const st = e.target.getStage()
                    if (st) st.container().style.cursor = 'grab'
                  }}
                  onMouseLeave={(e) => {
                    const st = e.target.getStage()
                    if (st) st.container().style.cursor = 'crosshair'
                  }}
                  onDragStart={(e) => {
                    e.cancelBubble = true
                    beginVertexDrag()
                  }}
                  onDragMove={(e) => {
                    // 边拖边更新：否则 Line 要等松手才跟上，看着像断开的
                    const nx = Math.min(1, Math.max(0, e.target.x() / imgW))
                    const ny = Math.min(1, Math.max(0, e.target.y() / imgH))
                    moveVertex(selected.id, i, nx, ny)
                  }}
                  onDragEnd={(e) => {
                    e.cancelBubble = true
                    endVertexDrag()
                  }}
                  onClick={(e) => {
                    e.cancelBubble = true
                  }}
                />
              ))}

              {/* 手画中的草稿：不受"显示标注"开关影响，正在画就得看得见 */}
              {draft && draft.length > 0 && (
                <Group listening={false}>
                  <Line
                    points={draft.flatMap(([x, y]) => [x * imgW, y * imgH])}
                    closed={draft.length >= 3}
                    stroke="#4ea1ff"
                    strokeWidth={2 * strokeUnit}
                    dash={[6 * strokeUnit, 4 * strokeUnit]}
                    fill={draft.length >= 3 ? 'rgba(78,161,255,0.18)' : undefined}
                  />
                  {draft.map(([x, y], i) => (
                    <Circle
                      key={i}
                      x={x * imgW}
                      y={y * imgH}
                      radius={(i === 0 ? 4.5 : 3.2) * strokeUnit}
                      fill={i === 0 ? '#4ea1ff' : '#fff'}
                      stroke="#4ea1ff"
                      strokeWidth={1.5 * strokeUnit}
                    />
                  ))}
                </Group>
              )}
            </Group>
          </Layer>
        </Stage>
      )}

      {/* 左上角：模式切换 */}
      {current && (
        <div className="stage-tools">
          <button
            className={`tool${mode === 'select' ? ' on' : ''}`}
            onClick={() => useStore.getState().setMode('select')}
            title="点选实例（V）"
          >
            选择
          </button>
          <button
            className={`tool${mode === 'refine' ? ' on' : ''}`}
            onClick={() => useStore.getState().setMode('refine')}
            title="点一下病斑，让模型在放大后的局部重新识别（P）"
          >
            点提示
          </button>
          <button
            className={`tool${mode === 'draw' ? ' on' : ''}`}
            onClick={() => useStore.getState().setMode('draw')}
            title="手画多边形：逐点单击，双击或点回起点闭合（D）"
          >
            手画
          </button>

          {mode === 'refine' && (
            <span className="tool-hint">
              在漏检的病斑上点一下
              {refining && ' · 识别中…'}
            </span>
          )}
          {mode === 'draw' && (
            <span className="tool-hint">
              {draft && draft.length > 0
                ? `${draft.length} 个点 · 双击闭合 · Backspace 退一点 · Esc 取消`
                : '逐点单击，≥3 点后双击闭合'}
            </span>
          )}

          <span className="tool-sep" />

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
        </div>
      )}

      {/* 覆盖层 */}
      {(!current || imgState === 'loading') && (
        <div className="stage-overlay">
          <div className="spinner" />
          <span>{current ? '加载图像…' : '从左侧选一张图'}</span>
        </div>
      )}
      {status === 'predicting' && (
        <div className="stage-badge predicting">
          <div className="spinner sm" /> 预标注中…
        </div>
      )}
      {refining && (
        <div className="stage-badge predicting">
          <div className="spinner sm" /> 点提示识别中…
        </div>
      )}
      {imgState === 'err' && (
        <div className="stage-overlay err">图像加载失败，检查后端是否在跑</div>
      )}
    </div>
  )
}

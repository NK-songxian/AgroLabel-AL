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

  /**
   * 点空白处才会走到这里 —— 实例的 Line 已经 cancelBubble 了。
   * refine 模式下：把屏幕坐标反算成图像像素坐标，交给点提示。
   */
  const onStageClick = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      const stage = stageRef.current
      const p = stage?.getPointerPosition()
      if (!p) return

      if (mode !== 'refine') {
        if (e.target === e.target.getStage()) select(null)
        return
      }
      if (refining || !imgW || !imgH) return

      const ix = (p.x - tx) / scale
      const iy = (p.y - ty) / scale
      // 点在图外不算
      if (ix < 0 || iy < 0 || ix > imgW || iy > imgH) return
      void refineAt(ix, iy)
    },
    [mode, refining, imgW, imgH, tx, ty, scale, select, refineAt],
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
          onDblClick={fit}
          style={{ cursor: mode === 'refine' ? 'copy' : 'crosshair' }}
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

              {instances.map((inst, i) => {
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

              {/* 选中实例的顶点，为后面的拖顶点微调留位 */}
              {selected?.polygon.map(([x, y], i) => (
                <Circle
                  key={i}
                  x={x * imgW}
                  y={y * imgH}
                  radius={3.2 * strokeUnit}
                  fill="#fff"
                  stroke={classStroke(selected.class_id)}
                  strokeWidth={1.4 * strokeUnit}
                  listening={false}
                />
              ))}
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
          {mode === 'refine' && (
            <span className="tool-hint">
              在漏检的病斑上点一下
              {refining && ' · 识别中…'}
            </span>
          )}
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

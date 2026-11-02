/**
 * Konva 画布：显示原图 + 叠加预标注多边形。
 *
 * 交互：
 *   - 滚轮缩放（以光标为锚点）、拖拽平移、双击 / F 自适应
 *   - 点多边形选中、点空白取消
 *   - 选择模式：拖顶点微调；**双击右键**在鼠标处插一个顶点
 *   - 点提示模式：点一下让模型重识别
 *   - 手画模式：逐点单击、双击闭合
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
  const draft = useStore((s) => s.draft)
  const drawAddPoint = useStore((s) => s.drawAddPoint)
  const drawCommit = useStore((s) => s.drawCommit)
  const beginVertexDrag = useStore((s) => s.beginVertexDrag)
  const moveVertex = useStore((s) => s.moveVertex)
  const endVertexDrag = useStore((s) => s.endVertexDrag)
  const insertVertexAt = useStore((s) => s.insertVertexAt)
  const toast = useStore((s) => s.toast)

  const [size, setSize] = useState({ width: 0, height: 0 })
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  const [imgState, setImgState] = useState<'idle' | 'loading' | 'ok' | 'err'>('idle')
  /** 上一次右键的时间与位置，用来识别「双击右键」（浏览器不给 button=2 的 dblclick） */
  const rclickRef = useRef<{ t: number; x: number; y: number } | null>(null)

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
      // Konva 的 click 不区分鼠标键：右键也会走这里，于是会把选中清掉，
      // 导致紧接着的"双击右键加节点"永远看不到 selectedId。主键之外一律忽略。
      if (e.evt?.button !== 0) return

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

  /** 右键双击间隔与位置容差 */
  const RCLICK_GAP_MS = 520
  const RCLICK_SLOP_PX = 10

  /**
   * 选择模式下**双击右键**：在鼠标位置插一个顶点。
   *
   * 浏览器不会为右键触发 dblclick（那是主键专有事件），所以自己配对：
   * 记录上一次右键的时间与位置，两次间隔够近就当成双击。
   * 顺带把原生右键菜单挡掉 —— 否则每插一个点弹一次菜单。
   */
  const onStageContextMenu = useCallback(
    (e: Konva.KonvaEventObject<PointerEvent>) => {
      e.evt.preventDefault()

      const p = stageRef.current?.getPointerPosition()
      if (!p) return
      const now = performance.now()
      const prev = rclickRef.current
      rclickRef.current = { t: now, x: p.x, y: p.y }

      if (mode !== 'select') return
      if (!prev || now - prev.t > RCLICK_GAP_MS) return
      if (Math.hypot(p.x - prev.x, p.y - prev.y) > RCLICK_SLOP_PX) return

      rclickRef.current = null
      if (!selectedId) {
        toast('先选中一个区域，再双击右键加节点', 'warn')
        return
      }
      const px = toImagePx(p)
      if (!px || !imgW || !imgH) return
      if (!insertVertexAt(px.x, px.y, imgW, imgH)) {
        toast('这个区域顶点太少，加不了节点', 'warn')
      }
    },
    [mode, selectedId, imgW, imgH, toImagePx, insertVertexAt, toast],
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
          onContextMenu={onStageContextMenu}
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

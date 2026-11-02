import { useEffect } from 'react'

import { useStore } from '../store'

/** 全局快捷键。焦点在输入框里时一律让行。 */
export function useHotkeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      const tag = t?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || t?.isContentEditable) return
      // antd Select 的下拉里按键不要被我们截走
      if (t?.closest?.('.ant-select')) return

      const s = useStore.getState()

      if ((e.key === 'z' || e.key === 'Z') && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        s.undo()
        return
      }
      switch (e.key) {
        case 'Delete':
        case 'Backspace':
          // 手画时 Backspace 是"退一个点"，不是删实例
          if (s.mode === 'draw' && s.draft?.length) {
            e.preventDefault()
            s.drawUndoPoint()
            break
          }
          if (s.selectedId) {
            e.preventDefault()
            s.deleteSelected()
          }
          break
        case 'Enter':
          e.preventDefault()
          s.confirmAll()
          break
        case 'Escape':
          // 手画中先取消草稿，再按一次才退出手画模式
          if (s.draft) {
            s.drawCancel()
            break
          }
          if (s.mode === 'draw') {
            s.setMode('select')
            break
          }
          s.select(null)
          break
        case 'ArrowRight':
          e.preventDefault()
          s.next()
          break
        case 'ArrowLeft':
          e.preventDefault()
          s.prev()
          break
        case 'f':
        case 'F':
          s.requestFit()
          break
        case 'v':
        case 'V':
          s.setMode('select')
          break
        case 'p':
        case 'P':
          s.setMode('refine')
          break
        case 'd':
        case 'D':
          s.setMode('draw')
          break
        case 'h':
        case 'H':
          s.toggleMasks()
          break
        default:
          break
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { Root } from './Root.tsx'
import { applyTheme, initialTheme } from './theme.ts'
import './styles.css'

// 先把 <html data-theme> 打上，避免首帧闪一下深色再变浅色。
// （index.html 里还有一段同步脚本更早执行，这里是兜底）
applyTheme(initialTheme())

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)

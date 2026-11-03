import { useMemo } from 'react'
import { ConfigProvider, theme } from 'antd'

import App from './App.tsx'
import { useStore } from './store.ts'

/**
 * antd 的两套 token，跟着 CSS 变量一起换。
 *
 * 光改 CSS 变量不够：antd 的 Select 下拉、气泡等是它自己算的色板，
 * 不换 algorithm 的话浅色模式下会冒出一块深色浮层。
 */
const ANTD_TOKENS = {
  dark: {
    colorPrimary: '#4ea1ff',
    colorBgContainer: '#171a21',
    colorBgElevated: '#1c2029',
    colorBorder: '#2a2f3a',
  },
  light: {
    colorPrimary: '#1c6fd8',
    colorBgContainer: '#ffffff',
    colorBgElevated: '#ffffff',
    colorBorder: '#dde2ea',
  },
} as const

/**
 * ConfigProvider 必须包在订阅主题的组件里 ——
 * 只有它能读到 store 的 theme，从而把 antd 的算法一起切换。
 */
export function Root() {
  const themeMode = useStore((s) => s.theme)
  const conf = useMemo(
    () => ({
      algorithm: themeMode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
      token: { ...ANTD_TOKENS[themeMode], borderRadius: 6, fontSize: 12 },
    }),
    [themeMode],
  )

  return (
    <ConfigProvider theme={conf}>
      <App />
    </ConfigProvider>
  )
}

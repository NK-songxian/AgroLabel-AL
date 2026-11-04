/**
 * 明暗主题：状态存在 localStorage，真实生效靠 `<html data-theme>`。
 *
 * 为什么不用 React state 驱动根节点：`styles.css` 里所有颜色都是 CSS 变量，
 * 只要改 `data-theme` 整站立刻换色，不需要组件重渲染 —— 画布上几百个多边形
 * 也不会因为换主题而重画。
 */

export type Theme = 'dark' | 'light'

export const THEME_KEY = 'agrolabel_theme'

/**
 * 默认白天：投影/投屏演示和日常办公环境下浅色更清楚。
 * 长时间盯画布想换深色，左下角一键切换，选择会记住。
 */
export const DEFAULT_THEME: Theme = 'light'

function isTheme(v: unknown): v is Theme {
  return v === 'dark' || v === 'light'
}

export function readStoredTheme(): Theme | null {
  try {
    const v = localStorage.getItem(THEME_KEY)
    return isTheme(v) ? v : null
  } catch {
    // 隐私模式 / 禁用存储：读不到就用默认值，不要抛
    return null
  }
}

export function initialTheme(): Theme {
  return readStoredTheme() ?? DEFAULT_THEME
}

/**
 * 应用主题并记住选择。
 *
 * `color-scheme` 也要设：它决定浏览器原生控件（滚动条、下拉箭头、
 * 输入框光标）是深还是浅，只改 CSS 变量不改它，浅色模式下滚动条还是黑的。
 */
export function applyTheme(t: Theme): void {
  const root = document.documentElement
  root.dataset.theme = t
  root.style.colorScheme = t
  try {
    localStorage.setItem(THEME_KEY, t)
  } catch {
    /* 存不了就算了，本次会话内照样生效 */
  }
}

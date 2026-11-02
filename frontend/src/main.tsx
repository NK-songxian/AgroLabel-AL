import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ConfigProvider, theme } from 'antd'

import App from './App.tsx'
import './styles.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#4ea1ff',
          borderRadius: 6,
          fontSize: 12,
          colorBgContainer: '#171a21',
          colorBgElevated: '#1c2029',
          colorBorder: '#2a2f3a',
        },
      }}
    >
      <App />
    </ConfigProvider>
  </StrictMode>,
)

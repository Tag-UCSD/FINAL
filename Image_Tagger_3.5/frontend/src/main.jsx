import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import { ToastProvider, MaintenanceOverlay } from '@/lib';
import '../index.css'
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ToastProvider>
      <MaintenanceOverlay />
      <App />
    </ToastProvider>
  </React.StrictMode>,
)
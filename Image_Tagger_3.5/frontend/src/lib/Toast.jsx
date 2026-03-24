import React, { createContext, useContext, useState, useCallback } from 'react';

/**
 * Simple global toast / notification system.
 *
 * Usage:
 *   import { ToastProvider, useToast } from '@shared';
 *
 *   // In main.jsx:
 *   ReactDOM.createRoot(...).render(
 *     <React.StrictMode>
 *       <ToastProvider>
 *         <App />
 *       </ToastProvider>
 *     </React.StrictMode>
 *   );
 *
 *   // In any child component:
 *   const toast = useToast();
 *   toast.success('Saved configuration.');
 *   toast.error('Upload failed.');
 */

const ToastContext = createContext(null);

export const ToastProvider = ({ children }) => {
  const [toasts, setToasts] = useState([]);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const addToast = useCallback((toast) => {
    const id = Math.random().toString(36).slice(2);
    const duration = toast.duration ?? 4000;

    const next = {
      id,
      type: toast.type || 'info',
      title: toast.title || null,
      message: toast.message || '',
    };

    setToasts((prev) => [...prev, next]);

    if (duration > 0) {
      setTimeout(() => {
        removeToast(id);
      }, duration);
    }
  }, [removeToast]);

  const api = {
    show: (opts) => addToast(opts),
    success: (message, opts = {}) => addToast({ type: 'success', message, ...opts }),
    error: (message, opts = {}) => addToast({ type: 'error', message, ...opts }),
    info: (message, opts = {}) => addToast({ type: 'info', message, ...opts }),
  };

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div className="fixed top-4 right-4 z-50 flex flex-col space-y-2 max-w-sm">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={
              'rounded-lg shadow-lg px-4 py-3 text-sm text-white flex items-start justify-between gap-3 ' +
              (t.type === 'success'
                ? 'bg-emerald-600'
                : t.type === 'error'
                  ? 'bg-red-600'
                  : 'bg-slate-800')
            }
          >
            <div className="flex-1">
              {t.title && <div className="font-semibold mb-0.5">{t.title}</div>}
              <div className="text-xs whitespace-pre-line">{t.message}</div>
            </div>
            <button
              type="button"
              onClick={() => removeToast(t.id)}
              className="ml-2 text-xs opacity-75 hover:opacity-100 focus:outline-none"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};

export const useToast = () => {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return ctx;
};

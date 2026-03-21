import React, { useEffect, useState } from 'react';

/**
 * Global full-screen maintenance overlay.
 *
 * This component listens for the custom "image-tagger:maintenance"
 * event emitted by the shared ApiClient whenever a 503 response is
 * returned from the backend. When triggered, it displays a clear,
 * student-friendly explanation that the system is temporarily
 * unavailable (e.g. DB down, migrations running, or heavy science
 * batch in progress).
 *
 * We intentionally centralise this behaviour in @shared so that all
 * frontends (Explorer, Workbench, Admin, Monitor) get the same UX
 * without duplicating code.
 */
export function MaintenanceOverlay() {
  const [active, setActive] = useState(false);
  const [lastMessage, setLastMessage] = useState('');
  const [lastEndpoint, setLastEndpoint] = useState('');

  useEffect(() => {
    function handler(event) {
      const detail = event.detail || {};
      setActive(true);
      setLastMessage(detail.message || 'The Image Tagger backend is temporarily unavailable.');
      setLastEndpoint(detail.endpoint || '');
    }

    if (typeof window !== 'undefined') {
      window.addEventListener('image-tagger:maintenance', handler);
    }
    return () => {
      if (typeof window !== 'undefined') {
        window.removeEventListener('image-tagger:maintenance', handler);
      }
    };
  }, []);

  if (!active) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60 backdrop-blur-sm">
      <div className="max-w-lg w-full mx-4 rounded-2xl bg-white shadow-2xl border border-red-100 p-6">
        <div className="flex items-start gap-3">
          <div className="mt-1 flex h-8 w-8 items-center justify-center rounded-full bg-red-50 border border-red-200">
            <span className="text-red-600 text-lg font-bold">!</span>
          </div>
          <div className="flex-1">
            <h2 className="text-lg font-semibold text-gray-900">
              Maintenance in progress
            </h2>
            <p className="mt-2 text-sm text-gray-700">
              One of the backend services is currently returning a{' '}
              <code className="px-1 py-0.5 text-xs bg-gray-100 rounded border border-gray-200">
                503 Service Unavailable
              </code>{' '}
              response. This usually means the database, science worker, or depth
              model is restarting or running a migration.
            </p>
            {lastEndpoint && (
              <p className="mt-2 text-xs text-gray-500 break-all">
                Last failing endpoint: <code>{lastEndpoint}</code>
              </p>
            )}
            {lastMessage && (
              <p className="mt-1 text-xs text-gray-500">
                Detail: {lastMessage}
              </p>
            )}
            <p className="mt-3 text-xs text-gray-600">
              You can safely leave this tab open. Once the backend is healthy
              again, try refreshing the page or re-running your last action.
            </p>
          </div>
        </div>
        <div className="mt-4 flex justify-end gap-3">
          <button
            type="button"
            onClick={() => setActive(false)}
            className="inline-flex items-center rounded-md border border-gray-300 bg-white px-3 py-1.5 text-xs font-medium text-gray-700 shadow-sm hover:bg-gray-50"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}

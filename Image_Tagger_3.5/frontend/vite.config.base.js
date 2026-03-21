import react from '@vitejs/plugin-react';
import path from 'path';

/**
 * Shared Vite Configuration
 * Ensures all 4 GUIs use the same aliases and build settings.
 */
export default function getBaseConfig(dirname) {
  const appName = path.basename(dirname);
  return {
    base: `/${appName}/`,
    plugins: [react()],
    resolve: {
      alias: {
        '@shared': path.resolve(dirname, '../../shared/src'),
        '@': path.resolve(dirname, './src'),
      },
    },
    server: {
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, '/v1'),
        },
        '/static': {
            target: 'http://127.0.0.1:8000',
            changeOrigin: true
        }
      },
    },
    build: {
        outDir: `../../dist/${appName}`,
        emptyOutDir: true
    }
  };
}
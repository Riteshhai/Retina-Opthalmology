import { fileURLToPath } from 'node:url'
import tailwind from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv, type Connect, type Plugin } from 'vite'

function restrictApi(): Plugin {
  const publicStatusOnly: Connect.NextHandleFunction = (request, response, next) => {
    const path = request.url?.split('?')[0] ?? ''
    if (/^\/(workspace|dist-workspace)(?:\/|%2f|$)/i.test(path)) {
      response.writeHead(404).end()
      return
    }
    if (!/^\/api(?:\/|%2f|$)/i.test(path)) {
      next()
      return
    }
    if (!/^\/api\/(health|ready)$/.test(path)) {
      response.writeHead(404).end()
      return
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      response.writeHead(405, { Allow: 'GET, HEAD' }).end()
      return
    }
    next()
  }
  return {
    name: 'qsart-public-api',
    configureServer(server) {
      server.middlewares.use(publicStatusOnly)
    },
    configurePreviewServer(server) {
      server.middlewares.use(publicStatusOnly)
    },
  }
}

export default defineConfig(({ mode }) => {
  if (mode === 'workspace') {
    return {
      root: fileURLToPath(new URL('./workspace', import.meta.url)),
      publicDir: false,
      plugins: [react()],
      build: { outDir: fileURLToPath(new URL('./dist-workspace', import.meta.url)), emptyOutDir: true },
    }
  }
  const env = loadEnv(mode, process.cwd(), 'QSART_')
  const target = env.QSART_API_TARGET || 'http://127.0.0.1:18000'
  return {
    plugins: [restrictApi(), react(), tailwind()],
    server: {
      fs: { deny: ['.env', '.env.*', '*.{crt,pem}', '**/.git/**', '**/workspace/**', '**/dist-workspace/**'] },
      proxy: {
        '^/api/(health|ready)(?:\\?.*)?$': {
          target,
          changeOrigin: true,
          rewrite: (path) => path.slice(4),
        },
      },
    },
  }
})

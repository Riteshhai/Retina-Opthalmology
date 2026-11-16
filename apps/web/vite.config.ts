import tailwind from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv, type Connect, type Plugin } from 'vite'

function restrictApi(): Plugin {
  const publicStatusOnly: Connect.NextHandleFunction = (request, response, next) => {
    const path = request.url?.split('?')[0] ?? ''
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
  const env = loadEnv(mode, process.cwd(), 'QSART_')
  const target = env.QSART_API_TARGET || 'http://127.0.0.1:18000'
  return {
    plugins: [restrictApi(), react(), tailwind()],
    server: {
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

/// <reference types="vitest/config" />
import { defineConfig, loadEnv, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Adds a Content-Security-Policy to the production page only (the dev server needs
// inline scripts for hot reload). Scripts may load only from this site, so an injected
// <script> would not run. Styles allow inline because React sets style attributes.
// The API origin is read from VITE_API_URL at build time.
function contentSecurityPolicy(apiUrl: string): Plugin {
  const api = (() => {
    try {
      return apiUrl ? new URL(apiUrl).origin : ''
    } catch {
      return ''
    }
  })()
  const policy = [
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src https://fonts.gstatic.com",
    "img-src 'self' data:",
    `connect-src 'self'${api ? ` ${api}` : ''}`,
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ].join('; ')
  return {
    name: 'content-security-policy',
    apply: 'build',
    transformIndexHtml: () => [
      { tag: 'meta', attrs: { 'http-equiv': 'Content-Security-Policy', content: policy }, injectTo: 'head-prepend' },
    ],
  }
}

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    tailwindcss(),
    contentSecurityPolicy(loadEnv(mode, process.cwd(), '').VITE_API_URL ?? ''),
  ],
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.{ts,tsx}'],
  },
}))

import { defineConfig } from 'vite'
import path from 'path'

export default defineConfig({
  build: {
    lib: {
      entry: {
        main: path.resolve(__dirname, 'src/main/main.ts'),
        preload: path.resolve(__dirname, 'src/preload/preload.ts')
      },
      formats: ['es']
    },
    outDir: 'dist-electron',
    minify: false,
    rollupOptions: {
      external: ['electron'],
      output: {
        entryFileNames: '[name]/[name].js',
        format: 'es'
      }
    }
  }
})

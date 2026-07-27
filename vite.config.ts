import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, open: true },
  resolve: {
    alias: {
      '@components': '/src/components',
      '@context': '/src/context',
      '@services': '/src/services',
      '@styles': '/src/styles'
    }
  }
});

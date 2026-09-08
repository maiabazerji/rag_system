import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendTarget = process.env.VITE_BACKEND_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // A bind mount from a Windows or macOS host does not forward inotify events
    // into the container, so Vite never learns that a file changed: it goes on
    // serving the module it transformed at startup, and every edit silently
    // fails to reach the browser even across a hard refresh. Polling is the
    // only thing that sees these writes.
    watch: { usePolling: true, interval: 300 },
    proxy: { "/api": { target: backendTarget, rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});

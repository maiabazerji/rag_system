/**
 * The static demo: the build Netlify serves (`npm run build:static`, i.e.
 * `vite build --mode static`). There is no backend, so the Advisor runs in the
 * browser (src/advisor-engine) and the Lab pages, which need the API, are hidden.
 *
 * Read at call time, not import time, so tests can switch it with `vi.stubEnv`.
 */
export function isStaticDemo(): boolean {
  return import.meta.env.VITE_STATIC_DEMO === "1" || import.meta.env.MODE === "static";
}

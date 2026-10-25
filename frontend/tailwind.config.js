/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // One family everywhere: titles, body text, labels and figures all use
      // Inter. `display` and `mono` stay as aliases so existing classes keep
      // working; figures line up through tabular-nums, not a monospace font.
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        bg: {
          DEFAULT: "#0d0e10",
          surface: "#16171b",
          elevated: "#1c1e23",
          border: "#2a2c33",
        },
        accent: {
          DEFAULT: "#e8a93c",
          hover: "#f1bd5d",
          soft: "#3a2f12",
        },
      },
      keyframes: {
        "fade-in": { from: { opacity: 0, transform: "translateY(4px)" }, to: { opacity: 1, transform: "none" } },
        shimmer: { "0%": { backgroundPosition: "-200% 0" }, "100%": { backgroundPosition: "200% 0" } },
      },
      animation: {
        "fade-in": "fade-in 0.25s ease-out",
        shimmer: "shimmer 1.4s linear infinite",
      },
    },
  },
  plugins: [],
};

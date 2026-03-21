/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "../../shared/src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        'enterprise-blue': '#0f172a',
        'action-primary': '#3b82f6',
        'surface-dark': '#1e293b'
      }
    },
  },
  plugins: [],
}

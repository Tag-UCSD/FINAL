/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
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

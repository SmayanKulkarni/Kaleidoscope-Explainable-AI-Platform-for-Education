/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        "on-primary": "#003730", 
        "secondary-container": "#1c5148", 
        "tertiary-container": "#e8a37c", 
        "background": "#0e1513", 
        "surface-tint": "#38ddc4", 
        "outline-variant": "#3b4a46", 
        "primary-fixed": "#5ffae0", 
        "on-secondary-container": "#8ec3b7", 
        "surface-variant": "#2f3634", 
        "on-tertiary-fixed-variant": "#6b3a1b", 
        "tertiary": "#ffc19f", 
        "on-background": "#dde4e1", 
        "secondary": "#9cd1c5", 
        "primary-fixed-dim": "#38ddc4", 
        "on-surface-variant": "#bacac5", 
        "error": "#ffb4ab", 
        "inverse-surface": "#dde4e1", 
        "surface-container-lowest": "#09100e", 
        "on-secondary-fixed": "#00201b", 
        "tertiary-fixed-dim": "#feb68e", 
        "on-error": "#690005", 
        "primary": "#44e5cc", 
        "inverse-primary": "#006b5d", 
        "surface": "#0e1513", 
        "surface-container-low": "#161d1b", 
        "on-primary-fixed": "#00201b", 
        "surface-container-highest": "#2f3634", 
        "on-secondary": "#003730", 
        "primary-container": "#00c9b1", 
        "error-container": "#93000a", 
        "tertiary-fixed": "#ffdbc9", 
        "surface-container-high": "#242b29", 
        "on-tertiary-fixed": "#331200", 
        "on-surface": "#dde4e1", 
        "on-tertiary": "#502407", 
        "outline": "#85948f", 
        "inverse-on-surface": "#2b3230", 
        "secondary-fixed": "#b7ede1", 
        "surface-container": "#1a211f", 
        "on-error-container": "#ffdad6", 
        "on-primary-container": "#004f44", 
        "on-secondary-fixed-variant": "#194f46", 
        "on-tertiary-container": "#69381a", 
        "on-primary-fixed-variant": "#005046", 
        "secondary-fixed-dim": "#9cd1c5", 
        "surface-dim": "#0e1513", 
        "surface-bright": "#333b39"
      },
      borderRadius: {
        DEFAULT: "0.25rem", lg: "0.5rem", xl: "0.75rem", full: "9999px",
        '2xl': '1rem', '3xl': '1.5rem'
      },
      fontFamily: {
        headline: ["Manrope", "sans-serif"], 
        body: ["Inter", "sans-serif"], 
        label: ["Space Grotesk", "sans-serif"], 
        display: ["Manrope", "sans-serif"]
      }
    }
  },
  plugins: [],
}

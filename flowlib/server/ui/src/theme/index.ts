/**
 * Flowlib Design System Theme
 * Single source of truth for all design tokens.
 */

export * from './colors'
export * from './typography'
export * from './spacing'
export * from './breakpoints'

import { lightTheme, darkTheme, hslToString, type ColorToken } from './colors'
import { fontFamilies } from './typography'

/**
 * Theme configuration.
 */
export const theme = {
  colors: {
    light: lightTheme,
    dark: darkTheme,
  },
  fonts: fontFamilies,
  radius: {
    none: '0',
    sm: '0.125rem', // 2px
    md: '0.375rem', // 6px
    lg: '0.5rem', // 8px
    xl: '1rem', // 16px
    full: '9999px',
  },
  shadows: {
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
  },
  transitions: {
    fast: '150ms',
    normal: '200ms',
    slow: '300ms',
  },
} as const

/**
 * Generate CSS variables for a theme.
 */
export function generateThemeVariables(themeColors: typeof lightTheme): Record<string, string> {
  const vars: Record<string, string> = {}
  for (const [key, color] of Object.entries(themeColors)) {
    vars[`--${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`] = hslToString(color)
  }
  return vars
}

/**
 * Get CSS variable name for a color token.
 */
export function getColorVariable(token: ColorToken): string {
  return `--${token.replace(/([A-Z])/g, '-$1').toLowerCase()}`
}


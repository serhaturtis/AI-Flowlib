/**
 * Color system for Flowlib UI.
 * Uses HSL color space for better manipulation and theming.
 * All colors are defined as semantic tokens for maintainability.
 */

export const colors = {
  // Base colors (HSL values)
  white: { h: 0, s: 0, l: 100 },
  black: { h: 0, s: 0, l: 0 },

  // Primary brand color (blue)
  primary: {
    50: { h: 210, s: 100, l: 98 },
    100: { h: 210, s: 100, l: 95 },
    200: { h: 210, s: 100, l: 90 },
    300: { h: 210, s: 100, l: 80 },
    400: { h: 210, s: 100, l: 70 },
    500: { h: 210, s: 100, l: 60 }, // Base primary
    600: { h: 210, s: 100, l: 50 },
    700: { h: 210, s: 100, l: 40 },
    800: { h: 210, s: 100, l: 30 },
    900: { h: 210, s: 100, l: 20 },
    950: { h: 210, s: 100, l: 10 },
  },

  // Neutral grays
  gray: {
    50: { h: 0, s: 0, l: 98 },
    100: { h: 0, s: 0, l: 95 },
    200: { h: 0, s: 0, l: 90 },
    300: { h: 0, s: 0, l: 80 },
    400: { h: 0, s: 0, l: 65 },
    500: { h: 0, s: 0, l: 50 },
    600: { h: 0, s: 0, l: 40 },
    700: { h: 0, s: 0, l: 30 },
    800: { h: 0, s: 0, l: 20 },
    900: { h: 0, s: 0, l: 10 },
    950: { h: 0, s: 0, l: 5 },
  },

  // Semantic colors
  success: {
    50: { h: 142, s: 76, l: 98 },
    100: { h: 142, s: 76, l: 95 },
    500: { h: 142, s: 76, l: 50 },
    600: { h: 142, s: 76, l: 40 },
    700: { h: 142, s: 76, l: 30 },
  },

  error: {
    50: { h: 0, s: 84, l: 98 },
    100: { h: 0, s: 84, l: 95 },
    500: { h: 0, s: 84, l: 50 },
    600: { h: 0, s: 84, l: 40 },
    700: { h: 0, s: 84, l: 30 },
  },

  warning: {
    50: { h: 38, s: 92, l: 98 },
    100: { h: 38, s: 92, l: 95 },
    500: { h: 38, s: 92, l: 50 },
    600: { h: 38, s: 92, l: 40 },
    700: { h: 38, s: 92, l: 30 },
  },

  info: {
    50: { h: 199, s: 89, l: 98 },
    100: { h: 199, s: 89, l: 95 },
    500: { h: 199, s: 89, l: 50 },
    600: { h: 199, s: 89, l: 40 },
    700: { h: 199, s: 89, l: 30 },
  },
} as const

/**
 * Convert HSL object to CSS HSL string.
 */
export function hslToString(color: { h: number; s: number; l: number }): string {
  return `${color.h} ${color.s}% ${color.l}%`
}

/**
 * Light theme color mappings.
 */
export const lightTheme = {
  background: colors.white,
  foreground: colors.gray[950],
  card: colors.white,
  'card-foreground': colors.gray[950],
  popover: colors.white,
  'popover-foreground': colors.gray[950],
  primary: colors.primary[500],
  'primary-foreground': colors.white,
  secondary: colors.gray[100],
  'secondary-foreground': colors.gray[900],
  muted: colors.gray[100],
  'muted-foreground': colors.gray[600],
  accent: colors.gray[200],
  'accent-foreground': colors.gray[900],
  destructive: colors.error[500],
  'destructive-foreground': colors.white,
  border: colors.gray[300],
  input: colors.gray[300],
  ring: colors.primary[500],
} as const

/**
 * Dark theme color mappings.
 */
export const darkTheme = {
  background: colors.gray[950],
  foreground: colors.gray[50],
  card: colors.gray[900],
  'card-foreground': colors.gray[50],
  popover: colors.gray[900],
  'popover-foreground': colors.gray[50],
  primary: colors.primary[400],
  'primary-foreground': colors.gray[950],
  secondary: colors.gray[800],
  'secondary-foreground': colors.gray[50],
  muted: colors.gray[800],
  'muted-foreground': colors.gray[400],
  accent: colors.gray[700],
  'accent-foreground': colors.gray[50],
  destructive: colors.error[500],
  'destructive-foreground': colors.white,
  border: colors.gray[700],
  input: colors.gray[700],
  ring: colors.primary[400],
} as const

export type ColorToken = keyof typeof lightTheme


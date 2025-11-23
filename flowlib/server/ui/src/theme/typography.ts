/**
 * Typography system for Flowlib UI.
 * Defines font families, sizes, weights, and line heights.
 */

export const fontFamilies = {
  sans: [
    '-apple-system',
    'BlinkMacSystemFont',
    '"Segoe UI"',
    'Roboto',
    '"Helvetica Neue"',
    'Arial',
    'sans-serif',
  ].join(', '),
  mono: [
    '"SF Mono"',
    'Monaco',
    '"Cascadia Code"',
    '"Roboto Mono"',
    'Consolas',
    '"Courier New"',
    'monospace',
  ].join(', '),
} as const

export const fontSizes = {
  xs: { size: '0.75rem', lineHeight: '1rem' }, // 12px / 16px
  sm: { size: '0.875rem', lineHeight: '1.25rem' }, // 14px / 20px
  base: { size: '1rem', lineHeight: '1.5rem' }, // 16px / 24px
  lg: { size: '1.125rem', lineHeight: '1.75rem' }, // 18px / 28px
  xl: { size: '1.25rem', lineHeight: '1.75rem' }, // 20px / 28px
  '2xl': { size: '1.5rem', lineHeight: '2rem' }, // 24px / 32px
  '3xl': { size: '1.875rem', lineHeight: '2.25rem' }, // 30px / 36px
  '4xl': { size: '2.25rem', lineHeight: '2.5rem' }, // 36px / 40px
  '5xl': { size: '3rem', lineHeight: '1' }, // 48px / 48px
  '6xl': { size: '3.75rem', lineHeight: '1' }, // 60px / 60px
} as const

export const fontWeights = {
  normal: 400,
  medium: 500,
  semibold: 600,
  bold: 700,
} as const

export const letterSpacing = {
  tighter: '-0.05em',
  tight: '-0.025em',
  normal: '0em',
  wide: '0.025em',
  wider: '0.05em',
  widest: '0.1em',
} as const

/**
 * Typography scale for headings and body text.
 */
export const typography = {
  h1: {
    fontSize: fontSizes['4xl'].size,
    lineHeight: fontSizes['4xl'].lineHeight,
    fontWeight: fontWeights.bold,
    letterSpacing: letterSpacing.tight,
  },
  h2: {
    fontSize: fontSizes['3xl'].size,
    lineHeight: fontSizes['3xl'].lineHeight,
    fontWeight: fontWeights.bold,
    letterSpacing: letterSpacing.tight,
  },
  h3: {
    fontSize: fontSizes['2xl'].size,
    lineHeight: fontSizes['2xl'].lineHeight,
    fontWeight: fontWeights.semibold,
    letterSpacing: letterSpacing.normal,
  },
  h4: {
    fontSize: fontSizes.xl.size,
    lineHeight: fontSizes.xl.lineHeight,
    fontWeight: fontWeights.semibold,
    letterSpacing: letterSpacing.normal,
  },
  h5: {
    fontSize: fontSizes.lg.size,
    lineHeight: fontSizes.lg.lineHeight,
    fontWeight: fontWeights.medium,
    letterSpacing: letterSpacing.normal,
  },
  h6: {
    fontSize: fontSizes.base.size,
    lineHeight: fontSizes.base.lineHeight,
    fontWeight: fontWeights.medium,
    letterSpacing: letterSpacing.normal,
  },
  body: {
    fontSize: fontSizes.base.size,
    lineHeight: fontSizes.base.lineHeight,
    fontWeight: fontWeights.normal,
    letterSpacing: letterSpacing.normal,
  },
  small: {
    fontSize: fontSizes.sm.size,
    lineHeight: fontSizes.sm.lineHeight,
    fontWeight: fontWeights.normal,
    letterSpacing: letterSpacing.normal,
  },
  caption: {
    fontSize: fontSizes.xs.size,
    lineHeight: fontSizes.xs.lineHeight,
    fontWeight: fontWeights.normal,
    letterSpacing: letterSpacing.normal,
  },
} as const

export type FontSize = keyof typeof fontSizes
export type FontWeight = keyof typeof fontWeights
export type TypographyVariant = keyof typeof typography


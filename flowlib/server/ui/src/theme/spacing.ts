/**
 * Spacing system for Flowlib UI.
 * Based on 4px base unit for consistency.
 */

export const spacing = {
  0: '0',
  1: '0.25rem', // 4px
  2: '0.5rem', // 8px
  3: '0.75rem', // 12px
  4: '1rem', // 16px
  5: '1.25rem', // 20px
  6: '1.5rem', // 24px
  8: '2rem', // 32px
  10: '2.5rem', // 40px
  12: '3rem', // 48px
  16: '4rem', // 64px
  20: '5rem', // 80px
  24: '6rem', // 96px
  32: '8rem', // 128px
} as const

/**
 * Semantic spacing tokens for common use cases.
 */
export const semanticSpacing = {
  xs: spacing[1], // 4px - tight spacing
  sm: spacing[2], // 8px - small spacing
  md: spacing[4], // 16px - medium spacing (default)
  lg: spacing[6], // 24px - large spacing
  xl: spacing[8], // 32px - extra large spacing
  '2xl': spacing[12], // 48px - 2x large spacing
  '3xl': spacing[16], // 64px - 3x large spacing
} as const

export type Spacing = keyof typeof spacing
export type SemanticSpacing = keyof typeof semanticSpacing


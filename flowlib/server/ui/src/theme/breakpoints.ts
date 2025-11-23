/**
 * Breakpoint system for responsive design.
 * Mobile-first approach.
 */

export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
} as const

/**
 * Media query helpers.
 */
export const mediaQueries = {
  sm: `(min-width: ${breakpoints.sm})`,
  md: `(min-width: ${breakpoints.md})`,
  lg: `(min-width: ${breakpoints.lg})`,
  xl: `(min-width: ${breakpoints.xl})`,
  '2xl': `(min-width: ${breakpoints['2xl']})`,
} as const

/**
 * Device categories for responsive design.
 */
export const devices = {
  mobile: `(max-width: ${breakpoints.sm})`,
  tablet: `(min-width: ${breakpoints.sm}) and (max-width: ${breakpoints.lg})`,
  desktop: `(min-width: ${breakpoints.lg})`,
} as const

export type Breakpoint = keyof typeof breakpoints
export type Device = keyof typeof devices


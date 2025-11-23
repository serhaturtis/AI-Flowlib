import { type HTMLAttributes, forwardRef } from 'react'
import { cn } from '../../utils/cn'

export interface GridProps extends HTMLAttributes<HTMLDivElement> {
  cols?: 1 | 2 | 3 | 4 | 6 | 12
  gap?: 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  responsive?: {
    sm?: 1 | 2 | 3 | 4 | 6 | 12
    md?: 1 | 2 | 3 | 4 | 6 | 12
    lg?: 1 | 2 | 3 | 4 | 6 | 12
  }
}

const Grid = forwardRef<HTMLDivElement, GridProps>(
  ({ className, cols = 1, gap = 'md', responsive, ...props }, ref) => {
    const colsClasses = {
      1: 'grid-cols-1',
      2: 'grid-cols-2',
      3: 'grid-cols-3',
      4: 'grid-cols-4',
      6: 'grid-cols-6',
      12: 'grid-cols-12',
    }

    const gapClasses = {
      none: 'gap-0',
      xs: 'gap-1',
      sm: 'gap-2',
      md: 'gap-4',
      lg: 'gap-6',
      xl: 'gap-8',
    }

    // Map responsive breakpoints to Tailwind classes
    // Using explicit class names so Tailwind JIT can detect them
    const responsiveClasses = responsive
      ? [
          responsive.sm === 1 && 'sm:grid-cols-1',
          responsive.sm === 2 && 'sm:grid-cols-2',
          responsive.sm === 3 && 'sm:grid-cols-3',
          responsive.sm === 4 && 'sm:grid-cols-4',
          responsive.sm === 6 && 'sm:grid-cols-6',
          responsive.sm === 12 && 'sm:grid-cols-12',
          responsive.md === 1 && 'md:grid-cols-1',
          responsive.md === 2 && 'md:grid-cols-2',
          responsive.md === 3 && 'md:grid-cols-3',
          responsive.md === 4 && 'md:grid-cols-4',
          responsive.md === 6 && 'md:grid-cols-6',
          responsive.md === 12 && 'md:grid-cols-12',
          responsive.lg === 1 && 'lg:grid-cols-1',
          responsive.lg === 2 && 'lg:grid-cols-2',
          responsive.lg === 3 && 'lg:grid-cols-3',
          responsive.lg === 4 && 'lg:grid-cols-4',
          responsive.lg === 6 && 'lg:grid-cols-6',
          responsive.lg === 12 && 'lg:grid-cols-12',
        ]
          .filter(Boolean)
          .join(' ')
      : ''

    return (
      <div
        ref={ref}
        className={cn(
          'grid',
          colsClasses[cols],
          gapClasses[gap],
          responsiveClasses,
          className,
        )}
        {...props}
      />
    )
  },
)
Grid.displayName = 'Grid'

export { Grid }


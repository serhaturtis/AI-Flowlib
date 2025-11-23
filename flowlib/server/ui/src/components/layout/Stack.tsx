import { type HTMLAttributes, forwardRef } from 'react'
import { cn } from '../../utils/cn'

export interface StackProps extends HTMLAttributes<HTMLDivElement> {
  direction?: 'row' | 'column'
  spacing?: 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  align?: 'start' | 'center' | 'end' | 'stretch'
  justify?: 'start' | 'center' | 'end' | 'between' | 'around' | 'evenly'
}

const Stack = forwardRef<HTMLDivElement, StackProps>(
  ({ className, direction = 'column', spacing = 'md', align, justify, ...props }, ref) => {
    const spacingClasses = {
      none: 'gap-0',
      xs: 'gap-1',
      sm: 'gap-2',
      md: 'gap-4',
      lg: 'gap-6',
      xl: 'gap-8',
    }

    const alignClasses = {
      start: 'items-start',
      center: 'items-center',
      end: 'items-end',
      stretch: 'items-stretch',
    }

    const justifyClasses = {
      start: 'justify-start',
      center: 'justify-center',
      end: 'justify-end',
      between: 'justify-between',
      around: 'justify-around',
      evenly: 'justify-evenly',
    }

    return (
      <div
        ref={ref}
        className={cn(
          'flex',
          direction === 'row' ? 'flex-row' : 'flex-col',
          spacingClasses[spacing],
          align && alignClasses[align],
          justify && justifyClasses[justify],
          className,
        )}
        {...props}
      />
    )
  },
)
Stack.displayName = 'Stack'

export { Stack }


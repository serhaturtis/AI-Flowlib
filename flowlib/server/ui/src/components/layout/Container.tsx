import { type HTMLAttributes, forwardRef } from 'react'
import { cn } from '../../utils/cn'

export interface ContainerProps extends HTMLAttributes<HTMLDivElement> {
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full'
}

const Container = forwardRef<HTMLDivElement, ContainerProps>(
  ({ className, maxWidth = 'xl', ...props }, ref) => {
    const maxWidthClasses = {
      sm: 'max-w-screen-sm',
      md: 'max-w-screen-md',
      lg: 'max-w-screen-lg',
      xl: 'max-w-screen-xl',
      '2xl': 'max-w-screen-2xl',
      full: 'max-w-full',
    }

    return (
      <div
        ref={ref}
        className={cn('mx-auto w-full px-4 sm:px-6 lg:px-8', maxWidthClasses[maxWidth], className)}
        {...props}
      />
    )
  },
)
Container.displayName = 'Container'

export { Container }


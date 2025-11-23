import { type HTMLAttributes, forwardRef } from 'react'
import { cn } from '../../utils/cn'

export interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {}

const Skeleton = forwardRef<HTMLDivElement, SkeletonProps>(
  ({ className, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn('animate-pulse rounded-md bg-muted', className)}
        {...props}
      />
    )
  },
)
Skeleton.displayName = 'Skeleton'

export { Skeleton }


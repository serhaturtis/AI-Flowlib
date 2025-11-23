import { type HTMLAttributes } from 'react'
import { Separator } from '../ui/Separator'
import { cn } from '../../utils/cn'

export interface DividerProps extends HTMLAttributes<HTMLDivElement> {
  orientation?: 'horizontal' | 'vertical'
  label?: string
}

export function Divider({ className, orientation = 'horizontal', label, ...props }: DividerProps) {
  if (label && orientation === 'horizontal') {
    return (
      <div className={cn('relative flex items-center', className)} {...props}>
        <Separator className="absolute" />
        <div className="relative bg-background px-2 text-sm text-muted-foreground">
          {label}
        </div>
      </div>
    )
  }

  return <Separator orientation={orientation} className={className} {...props} />
}


import { type ReactNode, type HTMLAttributes, useId } from 'react'
import { cn } from '../../utils/cn'

export interface SplitPaneProps extends HTMLAttributes<HTMLDivElement> {
  left: ReactNode
  right: ReactNode
  leftWidth?: string
  rightMinWidth?: string
  gap?: 'none' | 'sm' | 'md' | 'lg'
  orientation?: 'horizontal' | 'vertical'
}

export function SplitPane({
  left,
  right,
  leftWidth = '320px',
  rightMinWidth = '600px',
  gap = 'md',
  orientation = 'horizontal',
  className,
  ...props
}: SplitPaneProps) {
  const gapClasses = {
    none: '',
    sm: 'gap-2',
    md: 'gap-4',
    lg: 'gap-6',
  }
  const id = useId()
  const leftPaneId = `split-pane-left-${id}`
  const rightPaneId = `split-pane-right-${id}`

  if (orientation === 'vertical') {
    return (
      <div className={cn('flex flex-col', gapClasses[gap], className)} {...props}>
        <div className="flex-shrink-0">{left}</div>
        <div className="flex-1 min-h-0">{right}</div>
      </div>
    )
  }

  return (
    <div
      className={cn('flex flex-col lg:flex-row min-h-0', gapClasses[gap], className)}
      style={
        {
          '--left-width': leftWidth,
          '--right-min-width': rightMinWidth,
        } as React.CSSProperties
      }
      {...props}
    >
      <div
        id={leftPaneId}
        className="flex-shrink-0 overflow-y-auto w-full lg:w-[var(--left-width)] min-h-[200px]"
      >
        <div className="lg:hidden border-b border-border p-2 text-sm font-medium text-muted-foreground bg-muted/20">
          Configuration List
        </div>
        {left}
      </div>
      <div
        id={rightPaneId}
        className="flex-1 overflow-y-auto w-full lg:min-w-[var(--right-min-width)] min-h-[400px]"
      >
        <div className="lg:hidden border-b border-border p-2 text-sm font-medium text-muted-foreground bg-muted/20">
          Details & Editor
        </div>
        {right}
      </div>
    </div>
  )
}


import { type ReactNode, type HTMLAttributes } from 'react'
import { Label } from '../ui/Label'
import { cn } from '../../utils/cn'

export interface FormFieldProps extends HTMLAttributes<HTMLDivElement> {
  label?: string
  required?: boolean
  error?: string
  description?: string
  children: ReactNode
}

export function FormField({
  label,
  required = false,
  error,
  description,
  children,
  className,
  ...props
}: FormFieldProps) {
  return (
    <div className={cn('space-y-2', className)} {...props}>
      {label && (
        <Label>
          {label}
          {required && <span className="text-destructive ml-1">*</span>}
        </Label>
      )}
      {description && <p className="text-sm text-muted-foreground">{description}</p>}
      {children}
      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  )
}


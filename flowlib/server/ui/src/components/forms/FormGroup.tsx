import { type ReactNode, type FieldsetHTMLAttributes } from 'react'
import { cn } from '../../utils/cn'

export interface FormGroupProps extends FieldsetHTMLAttributes<HTMLFieldSetElement> {
  children: ReactNode
  legend?: string
}

export function FormGroup({ legend, children, className, ...props }: FormGroupProps) {
  return (
    <fieldset className={cn('space-y-4', className)} {...props}>
      {legend && (
        <legend className="text-sm font-medium leading-none">{legend}</legend>
      )}
      {children}
    </fieldset>
  )
}


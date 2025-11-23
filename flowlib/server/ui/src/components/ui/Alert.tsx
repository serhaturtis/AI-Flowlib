import { type HTMLAttributes, forwardRef } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../utils/cn'

const alertVariants = cva(
  'relative w-full rounded-lg border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground',
  {
    variants: {
      variant: {
        default: 'bg-background text-foreground',
        destructive:
          'border-destructive/50 text-destructive dark:border-destructive [&>svg]:text-destructive',
        success:
          'border-green-500/50 text-green-700 dark:text-green-400 dark:border-green-500 [&>svg]:text-green-600 dark:[&>svg]:text-green-400',
        warning:
          'border-yellow-500/50 text-yellow-700 dark:text-yellow-400 dark:border-yellow-500 [&>svg]:text-yellow-600 dark:[&>svg]:text-yellow-400',
        info: 'border-blue-500/50 text-blue-700 dark:text-blue-400 dark:border-blue-500 [&>svg]:text-blue-600 dark:[&>svg]:text-blue-400',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
)

const Alert = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>>(
  ({ className, variant, ...props }, ref) => (
    <div
      ref={ref}
      role="alert"
      className={cn(alertVariants({ variant }), className)}
      {...props}
    />
  ),
)
Alert.displayName = 'Alert'

const AlertTitle = forwardRef<HTMLParagraphElement, HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h5
      ref={ref}
      className={cn('mb-1 font-medium leading-none tracking-tight', className)}
      {...props}
    />
  ),
)
AlertTitle.displayName = 'AlertTitle'

const AlertDescription = forwardRef<HTMLParagraphElement, HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn('text-sm [&_p]:leading-relaxed', className)}
      {...props}
    />
  ),
)
AlertDescription.displayName = 'AlertDescription'

// eslint-disable-next-line react-refresh/only-export-components
export { Alert, AlertTitle, AlertDescription, alertVariants }


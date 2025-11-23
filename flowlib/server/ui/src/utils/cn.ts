/**
 * Utility for merging class names.
 * Combines clsx and tailwind-merge for optimal className handling.
 */

import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Merge class names with Tailwind conflict resolution.
 * Later classes override earlier ones when conflicts occur.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}


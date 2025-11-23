/**
 * Get the badge variant for a given status.
 *
 * @param status - The status string
 * @returns The badge variant to use
 */
export function getStatusBadgeVariant(
  status: string,
): 'default' | 'success' | 'destructive' | 'warning' | 'secondary' {
  if (status === 'completed' || status === 'success') return 'success'
  if (status === 'failed' || status === 'error') return 'destructive'
  if (status === 'running') return 'default'
  if (status === 'pending' || status === 'queued') return 'warning'
  return 'secondary'
}

import { Clock, CheckCircle2 } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Alert, AlertDescription } from '../ui/Alert'
import { Stack } from '../layout/Stack'
import type { RunRecord } from '../../hooks/agents/useRunsPolling'
import { getStatusBadgeVariant } from '../../utils/statusBadge'

export interface RunDetailsProps {
  run: RunRecord | null
}

/**
 * Detailed information about a selected run.
 *
 * Features:
 * - Status display
 * - Start/finish timestamps
 * - Message display
 * - Null-safe rendering
 */
export function RunDetails({ run }: RunDetailsProps) {
  if (!run) {
    return null
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Run Details</CardTitle>
        <CardDescription>Detailed information about the selected run</CardDescription>
      </CardHeader>
      <CardContent>
        <Stack spacing="md">
          <div className="flex items-center gap-2">
            <span className="font-medium">Status:</span>
            <Badge variant={getStatusBadgeVariant(run.status)}>{run.status}</Badge>
          </div>

          {run.started_at && (
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm">
                <span className="font-medium">Started:</span>{' '}
                {new Date(run.started_at).toLocaleString(undefined, {
                  dateStyle: 'medium',
                  timeStyle: 'short',
                })}
              </span>
            </div>
          )}

          {run.finished_at && (
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm">
                <span className="font-medium">Finished:</span>{' '}
                {new Date(run.finished_at).toLocaleString(undefined, {
                  dateStyle: 'medium',
                  timeStyle: 'short',
                })}
              </span>
            </div>
          )}

          {run.message && (
            <Alert>
              <AlertDescription>{run.message}</AlertDescription>
            </Alert>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

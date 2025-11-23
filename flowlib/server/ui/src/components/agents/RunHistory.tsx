import { History } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import type { AgentRunStatusResponse } from '../../services/agents'

export interface RunHistoryProps {
  history: AgentRunStatusResponse[]
  maxDisplay?: number
}

const getStatusBadgeVariant = (
  status: string,
): 'default' | 'success' | 'destructive' | 'warning' | 'secondary' => {
  if (status === 'completed' || status === 'success') return 'success'
  if (status === 'failed' || status === 'error') return 'destructive'
  if (status === 'running') return 'default'
  if (status === 'pending' || status === 'queued') return 'warning'
  return 'secondary'
}

/**
 * Historical run records display.
 *
 * Features:
 * - Shows recent run history
 * - Status badges
 * - Timestamp display
 * - Scrollable list
 * - Empty state
 */
export function RunHistory({ history, maxDisplay = 10 }: RunHistoryProps) {
  const displayHistory = history.slice(0, maxDisplay)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <History className="h-5 w-5" />
          Recent Runs
        </CardTitle>
        <CardDescription>Historical run records</CardDescription>
      </CardHeader>
      <CardContent>
        {history.length === 0 ? (
          <div className="text-center py-8">
            <History className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground text-sm">No history yet.</p>
          </div>
        ) : (
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {displayHistory.map((h) => (
              <Card key={h.run_id} className="cursor-pointer hover:bg-accent transition-colors">
                <CardContent className="p-3">
                  <div className="flex items-center justify-between mb-2">
                    <code className="text-xs font-mono bg-muted px-2 py-1 rounded">
                      {h.run_id.slice(0, 8)}...
                    </code>
                    <Badge variant={getStatusBadgeVariant(h.status)}>{h.status}</Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {h.started_at ? new Date(h.started_at).toLocaleDateString() : '—'}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

import { Play, Square } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Button } from '../ui/Button'
import type { RunRecord } from '../../hooks/agents/useRunsPolling'
import type { UseAgentRunFormResult } from '../../hooks/agents/useAgentRunForm'
import { getStatusBadgeVariant } from '../../utils/statusBadge'

export interface ActiveRunsListProps {
  runs: Record<string, RunRecord>
  selectedRunId: string | null
  onSelectRun: (runId: string) => void
  stopMutation: UseAgentRunFormResult['stopMutation']
}

/**
 * List of currently active agent runs.
 *
 * Features:
 * - Displays all active runs
 * - Click to select a run
 * - Stop button for running runs
 * - Status badges with color coding
 * - Empty state
 */
export function ActiveRunsList({
  runs,
  selectedRunId,
  onSelectRun,
  stopMutation,
}: ActiveRunsListProps) {
  const runsList = Object.values(runs)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Active Runs</CardTitle>
        <CardDescription>Currently running agent executions</CardDescription>
      </CardHeader>
      <CardContent>
        {runsList.length === 0 ? (
          <div className="text-center py-8">
            <Play className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground text-sm">No runs started yet.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {runsList.map((run) => (
              <Card
                key={run.run_id}
                className={`cursor-pointer hover:bg-accent transition-colors ${
                  selectedRunId === run.run_id ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() => onSelectRun(run.run_id)}
              >
                <CardContent className="p-3">
                  <div className="flex items-center justify-between mb-2">
                    <code className="text-xs font-mono bg-muted px-2 py-1 rounded">
                      {run.run_id.slice(0, 8)}...
                    </code>
                    <Badge variant={getStatusBadgeVariant(run.status)}>{run.status}</Badge>
                  </div>
                  <div className="text-sm font-medium">{run.agent_config_name}</div>
                  <div className="flex items-center gap-2 mt-2">
                    <Badge variant="outline" className="text-xs">
                      {run.mode}
                    </Badge>
                    {run.status === 'running' && (
                      <Button
                        type="button"
                        variant="destructive"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation()
                          stopMutation.mutate(run.run_id)
                        }}
                        disabled={stopMutation.isPending}
                        className="ml-auto"
                      >
                        <Square className="h-3 w-3 mr-1" />
                        Stop
                      </Button>
                    )}
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

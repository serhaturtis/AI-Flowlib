import { Play, Square, X, Trash2 } from 'lucide-react'
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
  onRemoveRun?: (runId: string) => void
  onClearCompleted?: () => void
}

/**
 * List of currently active agent runs.
 *
 * Features:
 * - Displays all active runs
 * - Click to select a run
 * - Stop button for running runs
 * - Remove button for completed runs
 * - Clear all completed runs button
 * - Status badges with color coding
 * - Empty state
 */
export function ActiveRunsList({
  runs,
  selectedRunId,
  onSelectRun,
  stopMutation,
  onRemoveRun,
  onClearCompleted,
}: ActiveRunsListProps) {
  const runsList = Object.values(runs)
  const hasCompletedRuns = runsList.some(
    (r) => r.status === 'completed' || r.status === 'failed' || r.status === 'cancelled'
  )

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Active Runs</CardTitle>
            <CardDescription>Currently running agent executions</CardDescription>
          </div>
          {hasCompletedRuns && onClearCompleted && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onClearCompleted}
              className="text-muted-foreground hover:text-foreground"
            >
              <Trash2 className="h-4 w-4 mr-1" />
              Clear Done
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {runsList.length === 0 ? (
          <div className="text-center py-8">
            <Play className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground text-sm">No runs started yet.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {runsList.map((run) => {
              const isFinished =
                run.status === 'completed' || run.status === 'failed' || run.status === 'cancelled'

              return (
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
                      <div className="flex items-center gap-1">
                        <Badge variant={getStatusBadgeVariant(run.status)}>{run.status}</Badge>
                        {isFinished && onRemoveRun && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
                            onClick={(e) => {
                              e.stopPropagation()
                              onRemoveRun(run.run_id)
                            }}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
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
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

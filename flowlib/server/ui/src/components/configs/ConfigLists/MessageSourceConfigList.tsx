import { UseQueryResult } from '@tanstack/react-query'
import { Timer, AlertCircle, Plus } from 'lucide-react'
import { MessageSourceSummary, MessageSourceListResponse } from '../../../services/configs'
import { Card } from '../../ui/Card'
import { Alert, AlertDescription } from '../../ui/Alert'
import { Badge } from '../../ui/Badge'
import { Skeleton } from '../../ui/Skeleton'
import { Button } from '../../ui/Button'

interface MessageSourceConfigListProps {
  messageSourcesQuery: UseQueryResult<MessageSourceListResponse, Error>
  onSelectSource: (config: MessageSourceSummary) => void
  onCreateSource: () => void
  selectedProject: string
}

/**
 * Displays list of message source configurations with loading/error states.
 * Shows source name, type, and enabled status.
 */
export function MessageSourceConfigList({
  messageSourcesQuery,
  onSelectSource,
  onCreateSource,
  selectedProject,
}: MessageSourceConfigListProps) {
  if (messageSourcesQuery.isLoading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    )
  }

  if (messageSourcesQuery.isError) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{messageSourcesQuery.error.message}</AlertDescription>
      </Alert>
    )
  }

  if (!messageSourcesQuery.data?.sources.length) {
    return (
      <div className="text-center py-8">
        <Timer className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <p className="text-muted-foreground text-sm">No message sources configured.</p>
        <p className="text-muted-foreground text-xs mt-1">
          Message sources trigger agents in daemon mode.
        </p>
        <Button size="sm" className="mt-4" disabled={!selectedProject} onClick={onCreateSource}>
          <Plus className="h-4 w-4 mr-1" />
          Create Source
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {messageSourcesQuery.data.sources.map((source) => (
        <Card
          key={source.name}
          className="cursor-pointer hover:bg-accent transition-colors p-3"
          onClick={() => onSelectSource(source)}
        >
          <div className="flex items-center justify-between">
            <div className="font-medium">{source.name}</div>
            {!source.enabled && (
              <Badge variant="secondary" className="text-xs">
                disabled
              </Badge>
            )}
          </div>
          <div className="flex gap-2 mt-2">
            <Badge variant="outline" className="text-xs">
              {source.source_type}
            </Badge>
            {source.source_type === 'timer' && typeof source.settings.interval_seconds === 'number' && (
              <Badge variant="secondary" className="text-xs">
                every {source.settings.interval_seconds}s
              </Badge>
            )}
          </div>
        </Card>
      ))}
    </div>
  )
}

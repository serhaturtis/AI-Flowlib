import { useEffect, useMemo } from 'react'
import { Activity, AlertCircle, CheckCircle2, XCircle, Radio, Loader2 } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Alert, AlertDescription } from '../ui/Alert'
import { Stack } from '../layout/Stack'
import { Button } from '../ui/Button'
import { useRunEvents, RunEvent, RunEventType, ConnectionState } from '../../hooks/agents/useRunEvents'

export interface RunEventStreamProps {
  /** Run ID to stream events from. Pass null to disconnect. */
  runId: string | null
  /** Current run status from polling (to know when to connect) */
  runStatus?: string
}

/**
 * Get icon and color for an event type
 */
function getEventTypeStyle(eventType: RunEventType): {
  icon: React.ReactNode
  color: string
  label: string
} {
  switch (eventType) {
    case 'run_started':
      return {
        icon: <Radio className="h-3 w-3" />,
        color: 'text-blue-500',
        label: 'Started',
      }
    case 'run_completed':
      return {
        icon: <CheckCircle2 className="h-3 w-3" />,
        color: 'text-green-500',
        label: 'Completed',
      }
    case 'run_failed':
      return {
        icon: <XCircle className="h-3 w-3" />,
        color: 'text-red-500',
        label: 'Failed',
      }
    case 'run_cancelled':
      return {
        icon: <XCircle className="h-3 w-3" />,
        color: 'text-yellow-500',
        label: 'Cancelled',
      }
    case 'activity':
      return {
        icon: <Activity className="h-3 w-3" />,
        color: 'text-purple-500',
        label: 'Activity',
      }
    case 'cycle_started':
    case 'cycle_completed':
      return {
        icon: <Loader2 className="h-3 w-3" />,
        color: 'text-cyan-500',
        label: 'Cycle',
      }
    case 'log':
      return {
        icon: <Activity className="h-3 w-3" />,
        color: 'text-gray-500',
        label: 'Log',
      }
    case 'error':
      return {
        icon: <AlertCircle className="h-3 w-3" />,
        color: 'text-red-500',
        label: 'Error',
      }
    default:
      return {
        icon: <Activity className="h-3 w-3" />,
        color: 'text-gray-500',
        label: eventType,
      }
  }
}

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp)
    const time = date.toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
    // Add milliseconds manually
    const ms = date.getMilliseconds().toString().padStart(3, '0')
    return `${time}.${ms}`
  } catch {
    return timestamp
  }
}

/**
 * Get the main message from an event
 */
function getEventMessage(event: RunEvent): string {
  const data = event.data as Record<string, unknown>

  // For activity events, use the formatted or message field
  if (event.event_type === 'activity') {
    return (data.formatted as string) || (data.message as string) || 'Activity'
  }

  // For lifecycle events, use the message or status
  if (data.message) {
    return data.message as string
  }

  if (data.status) {
    return `Status: ${data.status}`
  }

  // For cycle events
  if (data.cycle_number) {
    const max = data.max_cycles ? `/${data.max_cycles}` : ''
    return `Cycle ${data.cycle_number}${max}`
  }

  return event.event_type
}

/**
 * Connection status indicator
 */
function ConnectionStatus({ state, error }: { state: ConnectionState; error: string | null }) {
  const getStatusStyle = () => {
    switch (state) {
      case 'connected':
        return { variant: 'success' as const, text: 'Connected' }
      case 'connecting':
        return { variant: 'secondary' as const, text: 'Connecting...' }
      case 'error':
        return { variant: 'destructive' as const, text: 'Error' }
      default:
        return { variant: 'outline' as const, text: 'Disconnected' }
    }
  }

  const { variant, text } = getStatusStyle()

  return (
    <div className="flex items-center gap-2">
      <Badge variant={variant}>{text}</Badge>
      {error && <span className="text-xs text-red-500">{error}</span>}
    </div>
  )
}

/**
 * Single event item in the stream
 */
function EventItem({ event }: { event: RunEvent }) {
  const style = getEventTypeStyle(event.event_type)
  const message = getEventMessage(event)

  return (
    <div className="flex items-start gap-2 py-1 px-2 hover:bg-muted/50 rounded text-sm font-mono">
      <span className="text-muted-foreground shrink-0 text-xs">{formatTimestamp(event.timestamp)}</span>
      <span className={`shrink-0 ${style.color}`}>{style.icon}</span>
      <span className="whitespace-pre-wrap break-all">{message}</span>
    </div>
  )
}

/**
 * Real-time event stream display for an agent run.
 *
 * Features:
 * - Auto-connects when run is in "running" status
 * - Displays streaming events in a scrollable log
 * - Shows connection status
 * - Auto-scrolls to latest events
 * - Handles reconnection on disconnect
 */
export function RunEventStream({ runId, runStatus }: RunEventStreamProps) {
  const { events, connectionState, error, isStreaming, logRef, connect, disconnect, clearEvents } =
    useRunEvents()

  // Auto-connect when run is running
  useEffect(() => {
    if (runId && runStatus === 'running') {
      connect(runId)
    } else {
      disconnect()
    }

    return () => {
      disconnect()
    }
  }, [runId, runStatus, connect, disconnect])

  // Group events by type for summary
  const eventSummary = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const event of events) {
      counts[event.event_type] = (counts[event.event_type] || 0) + 1
    }
    return counts
  }, [events])

  // Show placeholder when no run selected or not streaming
  if (!runId || (!isStreaming && events.length === 0 && runStatus !== 'running')) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Event Stream
          </CardTitle>
          <CardDescription>
            Real-time activity events will appear here when the run is active
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            {!runId
              ? 'Select a run to view its event stream'
              : runStatus === 'pending'
                ? 'Waiting for run to start...'
                : runStatus === 'completed' || runStatus === 'failed' || runStatus === 'cancelled'
                  ? 'Run has ended. No live events available.'
                  : 'Select a running agent to view its event stream'}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Event Stream
          </CardTitle>
          <ConnectionStatus state={connectionState} error={error} />
        </div>
        <CardDescription className="flex items-center gap-4">
          <span>{events.length} events</span>
          {Object.entries(eventSummary).length > 0 && (
            <span className="text-xs">
              {Object.entries(eventSummary)
                .map(([type, count]) => `${type}: ${count}`)
                .join(' | ')}
            </span>
          )}
          {events.length > 0 && (
            <Button variant="ghost" size="sm" onClick={clearEvents} className="h-6 text-xs">
              Clear
            </Button>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Stack spacing="sm">
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div
            ref={logRef}
            className="h-[400px] overflow-y-auto border rounded-md bg-muted/30 p-2"
            style={{ fontFamily: 'monospace' }}
          >
            {events.length === 0 ? (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                {isStreaming ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Waiting for events...
                  </span>
                ) : (
                  'No events yet'
                )}
              </div>
            ) : (
              events.map((event) => <EventItem key={event.event_id} event={event} />)
            )}
          </div>
        </Stack>
      </CardContent>
    </Card>
  )
}

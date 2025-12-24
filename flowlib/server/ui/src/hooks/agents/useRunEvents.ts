import { useEffect, useRef, useState, useCallback } from 'react'
import { POLLING_INTERVALS } from '../../constants/polling'

/**
 * Event types emitted during agent execution.
 * Maps to RunEventType enum on the server.
 */
export type RunEventType =
  | 'activity'
  | 'run_started'
  | 'run_completed'
  | 'run_failed'
  | 'run_cancelled'
  | 'cycle_started'
  | 'cycle_completed'
  | 'log'
  | 'error'

/**
 * A single event from an agent run.
 */
export interface RunEvent {
  event_id: string
  run_id: string
  event_type: RunEventType
  timestamp: string
  data: Record<string, unknown>
}

/**
 * Connection state for the WebSocket.
 */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error'

/**
 * Result returned by the useRunEvents hook.
 */
export interface UseRunEventsResult {
  /** List of events received from the run */
  events: RunEvent[]
  /** Current WebSocket connection state */
  connectionState: ConnectionState
  /** Error message if connection failed */
  error: string | null
  /** Whether we're currently connected and receiving events */
  isStreaming: boolean
  /** Ref for the log container (for auto-scroll) */
  logRef: React.RefObject<HTMLDivElement>
  /** Connect to a run's event stream */
  connect: (runId: string) => void
  /** Disconnect from the current event stream */
  disconnect: () => void
  /** Clear all received events */
  clearEvents: () => void
}

/**
 * Hook for streaming real-time events from an agent run via WebSocket.
 *
 * Features:
 * - Connects to WebSocket endpoint for run events
 * - Auto-reconnect with exponential backoff
 * - Event buffering with memory cap
 * - Auto-scroll to latest events
 * - Proper cleanup on unmount or disconnect
 *
 * @example
 * ```tsx
 * function RunMonitor({ runId }: { runId: string }) {
 *   const { events, isStreaming, connect, disconnect, logRef } = useRunEvents()
 *
 *   useEffect(() => {
 *     if (runId) {
 *       connect(runId)
 *     }
 *     return () => disconnect()
 *   }, [runId])
 *
 *   return (
 *     <div ref={logRef}>
 *       {events.map(e => <EventItem key={e.event_id} event={e} />)}
 *     </div>
 *   )
 * }
 * ```
 */
export function useRunEvents(): UseRunEventsResult {
  const [events, setEvents] = useState<RunEvent[]>([])
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected')
  const [error, setError] = useState<string | null>(null)

  const websocketRef = useRef<WebSocket | null>(null)
  const logRef = useRef<HTMLDivElement | null>(null)
  const reconnectTimerRef = useRef<number | null>(null)
  const isMountedRef = useRef<boolean>(true)
  const reconnectAttemptsRef = useRef<number>(0)
  const currentRunIdRef = useRef<string | null>(null)

  // Configuration
  const MAX_RECONNECT_ATTEMPTS = 5
  const INITIAL_RECONNECT_DELAY = POLLING_INTERVALS.WEBSOCKET_RECONNECT ?? 1000
  const MAX_RECONNECT_DELAY = 10000 // 10 seconds
  const MAX_EVENT_HISTORY = 2000 // Maximum events to keep in memory

  /**
   * Clean up WebSocket and timers
   */
  const cleanup = useCallback(() => {
    // Clear reconnection timer
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }

    // Close WebSocket with proper cleanup
    if (websocketRef.current) {
      const ws = websocketRef.current
      // Remove handlers to prevent callbacks
      ws.onclose = null
      ws.onerror = null
      ws.onmessage = null
      ws.onopen = null
      ws.close()
      websocketRef.current = null
    }
  }, [])

  /**
   * Connect to WebSocket for run event streaming
   */
  const connectWebSocket = useCallback(
    (runId: string) => {
      // Clean up any existing connection
      cleanup()

      setConnectionState('connecting')

      const wsUrl = `${window.location.origin.replace(/^http/, 'ws')}/api/v1/agents/runs/${runId}/events`
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        if (isMountedRef.current) {
          setConnectionState('connected')
          setError(null)
          reconnectAttemptsRef.current = 0
        }
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          // Check for error responses from server
          if (data.error) {
            setError(data.message || 'Stream error')
            setConnectionState('error')
            return
          }

          // Add event to buffer
          setEvents((prev) => {
            const next = [...prev, data as RunEvent]
            // Cap to avoid memory growth
            return next.length > MAX_EVENT_HISTORY ? next.slice(-MAX_EVENT_HISTORY) : next
          })
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err)
        }
      }

      ws.onerror = (err) => {
        console.error('WebSocket error:', err)
        if (isMountedRef.current) {
          setError('Connection error')
          setConnectionState('error')
        }
      }

      ws.onclose = (event) => {
        if (!isMountedRef.current) return

        setConnectionState('disconnected')
        websocketRef.current = null

        // Don't reconnect if this was a policy violation (server rejected connection)
        if (event.code === 1008) {
          setError('Stream ended or not available')
          return
        }

        // Attempt auto-reconnect if:
        // 1. Component is still mounted
        // 2. Run ID matches current run
        // 3. Haven't exceeded max reconnection attempts
        if (
          isMountedRef.current &&
          currentRunIdRef.current === runId &&
          reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS
        ) {
          reconnectAttemptsRef.current++

          // Calculate exponential backoff delay
          const delay = Math.min(
            INITIAL_RECONNECT_DELAY * Math.pow(2, reconnectAttemptsRef.current - 1),
            MAX_RECONNECT_DELAY
          )

          console.log(
            `Run event WebSocket closed. Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})...`
          )

          reconnectTimerRef.current = window.setTimeout(() => {
            if (isMountedRef.current && currentRunIdRef.current === runId) {
              connectWebSocket(runId)
            }
          }, delay) as unknown as number
        } else if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
          setError('Connection lost. Maximum reconnection attempts exceeded.')
          setConnectionState('error')
        }
      }

      websocketRef.current = ws
    },
    [cleanup, INITIAL_RECONNECT_DELAY, MAX_RECONNECT_ATTEMPTS, MAX_RECONNECT_DELAY, MAX_EVENT_HISTORY]
  )

  /**
   * Connect to a run's event stream
   */
  const connect = useCallback(
    (runId: string) => {
      if (!runId) {
        setError('Run ID is required')
        return
      }

      // Update current run ID ref
      currentRunIdRef.current = runId

      // Reset state for new connection
      setEvents([])
      setError(null)
      reconnectAttemptsRef.current = 0

      connectWebSocket(runId)
    },
    [connectWebSocket]
  )

  /**
   * Disconnect from the current event stream
   */
  const disconnect = useCallback(() => {
    currentRunIdRef.current = null
    cleanup()
    setConnectionState('disconnected')
    setError(null)
    reconnectAttemptsRef.current = 0
  }, [cleanup])

  /**
   * Clear all received events
   */
  const clearEvents = useCallback(() => {
    setEvents([])
  }, [])

  /**
   * Auto-scroll log to bottom when new events arrive
   */
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [events])

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      isMountedRef.current = false
      currentRunIdRef.current = null
      cleanup()
    }
  }, [cleanup])

  return {
    events,
    connectionState,
    error,
    isStreaming: connectionState === 'connected',
    logRef,
    connect,
    disconnect,
    clearEvents,
  }
}

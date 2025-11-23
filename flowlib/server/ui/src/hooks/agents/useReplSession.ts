import { useEffect, useRef, useState } from 'react'
import {
  createReplSession,
  closeReplSession,
  sendReplInput,
  ReplSessionResponse,
} from '../../services/agents'
import { POLLING_INTERVALS } from '../../constants/polling'

export interface ReplMessage {
  type: string
  content?: string
  message?: string
  ts?: string
}

export interface UseReplSessionResult {
  session: ReplSessionResponse | null
  messages: ReplMessage[]
  isConnected: boolean
  logRef: React.RefObject<HTMLDivElement>
  startSession: (projectId: string, agentName: string) => Promise<void>
  closeSession: () => Promise<void>
  sendMessage: (input: string) => Promise<void>
  clearMessages: () => void
  error: string | null
}

/**
 * Hook for managing REPL session lifecycle and WebSocket connection.
 *
 * Features:
 * - Creates/closes REPL sessions
 * - Manages WebSocket connection with auto-reconnect
 * - Handles message streaming
 * - Auto-scrolls log to bottom
 * - Cleans up connections on unmount
 *
 * @returns REPL session state and control methods
 */
export function useReplSession(): UseReplSessionResult {
  const [session, setSession] = useState<ReplSessionResponse | null>(null)
  const [messages, setMessages] = useState<ReplMessage[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const websocketRef = useRef<WebSocket | null>(null)
  const logRef = useRef<HTMLDivElement | null>(null)
  const reconnectTimerRef = useRef<number | null>(null)
  const isMountedRef = useRef<boolean>(true)
  const reconnectAttemptsRef = useRef<number>(0)
  const currentSessionIdRef = useRef<string | null>(null)

  // WebSocket reconnection configuration
  const MAX_RECONNECT_ATTEMPTS = 10
  const INITIAL_RECONNECT_DELAY = POLLING_INTERVALS.WEBSOCKET_RECONNECT
  const MAX_RECONNECT_DELAY = 30000 // 30 seconds
  const MAX_EVENT_HISTORY = 1000 // Maximum number of events to keep in memory

  /**
   * Connect to WebSocket for REPL event streaming with exponential backoff
   */
  const connectWebSocket = (sessionId: string) => {
    // Clear any pending reconnection timer
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }

    // Close existing connection properly
    if (websocketRef.current) {
      const oldWs = websocketRef.current
      // Remove event handlers to prevent triggering reconnection from old connection
      oldWs.onclose = null
      oldWs.onerror = null
      oldWs.onmessage = null
      oldWs.onopen = null
      oldWs.close()
      websocketRef.current = null
    }

    const wsUrl = `${window.location.origin.replace(/^http/, 'ws')}/api/v1/agents/repl/sessions/${sessionId}/events`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      setIsConnected(true)
      setError(null)
      // Reset reconnection attempts on successful connection
      reconnectAttemptsRef.current = 0
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setMessages((prev) => {
          const next = [...prev, { ...data, ts: new Date().toISOString() }]
          // Cap to avoid memory growth
          return next.length > MAX_EVENT_HISTORY ? next.slice(-MAX_EVENT_HISTORY) : next
        })
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err)
      }
    }

    ws.onerror = (err) => {
      console.error('WebSocket error:', err)
      setError('WebSocket connection error')
      setIsConnected(false)
    }

    ws.onclose = () => {
      setIsConnected(false)
      websocketRef.current = null

      // Attempt auto-reconnect only if:
      // 1. Component is still mounted
      // 2. Session ID matches the current session (use ref to avoid stale closure)
      // 3. Haven't exceeded max reconnection attempts
      if (
        isMountedRef.current &&
        currentSessionIdRef.current === sessionId &&
        reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS
      ) {
        reconnectAttemptsRef.current++

        // Calculate exponential backoff delay: initial * (2 ^ attempts)
        const delay = Math.min(
          INITIAL_RECONNECT_DELAY * Math.pow(2, reconnectAttemptsRef.current - 1),
          MAX_RECONNECT_DELAY
        )

        console.log(
          `WebSocket closed. Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})...`
        )

        reconnectTimerRef.current = window.setTimeout(() => {
          // Double-check still mounted and session matches before reconnecting
          if (isMountedRef.current && currentSessionIdRef.current === sessionId) {
            connectWebSocket(sessionId)
          }
        }, delay) as unknown as number
      } else if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
        console.error('Max WebSocket reconnection attempts reached. Please refresh or restart the session.')
        setError('Connection lost. Maximum reconnection attempts exceeded.')
      }
    }

    websocketRef.current = ws
  }

  /**
   * Start a new REPL session
   */
  const startSession = async (projectId: string, agentName: string) => {
    if (!projectId || !agentName) {
      setError('Project and agent must be selected to start a REPL session.')
      return
    }

    try {
      const newSession = await createReplSession(projectId, agentName)
      setSession(newSession)
      setMessages([])
      setError(null)
      // Update session ID ref before connecting to prevent stale closures
      currentSessionIdRef.current = newSession.session_id
      // Reset reconnection attempts for new session
      reconnectAttemptsRef.current = 0
      connectWebSocket(newSession.session_id)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start REPL session.'
      setError(message)
      throw err // Re-throw so caller can handle
    }
  }

  /**
   * Close the current REPL session
   */
  const closeSession = async () => {
    if (!session) return

    try {
      await closeReplSession(session.session_id)
    } catch (err) {
      console.error('Failed to close REPL session:', err)
    } finally {
      // Clear session ID ref to prevent reconnection attempts
      currentSessionIdRef.current = null

      // Clear reconnection timer
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }

      // Close WebSocket with proper cleanup
      if (websocketRef.current) {
        const ws = websocketRef.current
        // Remove handlers to prevent reconnection
        ws.onclose = null
        ws.onerror = null
        ws.onmessage = null
        ws.onopen = null
        ws.close()
        websocketRef.current = null
      }

      // Reset state
      setSession(null)
      setMessages([])
      setIsConnected(false)
      setError(null)
      reconnectAttemptsRef.current = 0
    }
  }

  /**
   * Send a message to the REPL session
   */
  const sendMessage = async (input: string) => {
    if (!session) {
      throw new Error('No active REPL session')
    }
    if (!input.trim()) {
      return
    }

    try {
      await sendReplInput(session.session_id, input)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send message.'
      setError(message)
      throw err
    }
  }

  /**
   * Clear all messages
   */
  const clearMessages = () => {
    setMessages([])
  }

  /**
   * Auto-scroll log to bottom when new messages arrive
   */
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [messages])

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      // Mark as unmounted to prevent reconnection attempts
      isMountedRef.current = false

      // Clear session ID ref
      currentSessionIdRef.current = null

      // Clear reconnection timer
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }

      // Close WebSocket
      if (websocketRef.current) {
        const ws = websocketRef.current
        // Remove handlers to prevent any callbacks
        ws.onclose = null
        ws.onerror = null
        ws.onmessage = null
        ws.onopen = null
        ws.close()
        websocketRef.current = null
      }
    }
  }, [])

  return {
    session,
    messages,
    isConnected,
    logRef,
    startSession,
    closeSession,
    sendMessage,
    clearMessages,
    error,
  }
}

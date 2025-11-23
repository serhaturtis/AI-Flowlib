import { useEffect, useState, useRef } from 'react'
import { listRuns, listRunHistory, getRunStatus, AgentRunStatusResponse } from '../../services/agents'
import { POLLING_INTERVALS } from '../../constants/polling'

export interface RunRecord {
  run_id: string
  status: string
  agent_config_name: string
  mode: string
  started_at: string | null
  finished_at: string | null
  message: string | null
}

export interface UseRunsPollingResult {
  runs: Record<string, RunRecord>
  history: AgentRunStatusResponse[]
  setRuns: React.Dispatch<React.SetStateAction<Record<string, RunRecord>>>
  addRun: (run: RunRecord) => void
  updateRun: (runId: string, updates: Partial<RunRecord>) => void
  pollingError: string | null
  isPollingPaused: boolean
}

/**
 * Hook for polling active runs and run history.
 *
 * Features:
 * - Polls active runs every 5 seconds
 * - Polls run history every 7 seconds
 * - Polls selected run status every 2 seconds (if selected)
 * - Provides methods to add/update runs
 *
 * @param selectedRunId - Currently selected run ID for focused polling
 * @param agentName - Agent name for fallback in status updates
 * @param mode - Execution mode for fallback in status updates
 * @returns Runs state, history, and update methods
 */
export function useRunsPolling(
  selectedRunId: string | null,
  agentName: string,
  mode: string,
): UseRunsPollingResult {
  const [runs, setRuns] = useState<Record<string, RunRecord>>({})
  const [history, setHistory] = useState<AgentRunStatusResponse[]>([])
  const [pollingError, setPollingError] = useState<string | null>(null)
  const [isPollingPaused, setIsPollingPaused] = useState(false)

  // Use refs to avoid triggering polling restart when agentName/mode change
  const agentNameRef = useRef(agentName)
  const modeRef = useRef(mode)

  // Error tracking for exponential backoff
  const consecutiveFailuresRef = useRef({
    runs: 0,
    history: 0,
    selectedRun: 0,
  })

  // Polling configuration
  const MAX_CONSECUTIVE_FAILURES = 5
  const BACKOFF_MULTIPLIER = 2
  const MAX_BACKOFF_DELAY = 60000 // 1 minute

  // Keep refs updated
  useEffect(() => {
    agentNameRef.current = agentName
    modeRef.current = mode
  }, [agentName, mode])

  /**
   * Poll all active runs with exponential backoff on failures
   */
  useEffect(() => {
    const intervalId: NodeJS.Timeout | null = null
    let timeoutId: NodeJS.Timeout | null = null

    const loadRuns = async () => {
      // Skip if polling is paused due to max failures
      if (isPollingPaused) {
        return
      }

      try {
        const existing = await listRuns()

        // Success: reset failure counter and clear errors
        consecutiveFailuresRef.current.runs = 0
        setPollingError(null)

        setRuns((prev) => {
          const next = { ...prev }
          existing.forEach((r) => {
            next[r.run_id] = {
              run_id: r.run_id,
              status: r.status,
              // Preserve agent name and mode from previous state if available
              agent_config_name: prev[r.run_id]?.agent_config_name ?? '',
              mode: prev[r.run_id]?.mode ?? '',
              started_at: r.started_at,
              finished_at: r.finished_at,
              message: r.message,
            }
          })
          return next
        })

        // Schedule next poll at normal interval
        scheduleNextPoll(POLLING_INTERVALS.ACTIVE_RUNS)
      } catch (error) {
        consecutiveFailuresRef.current.runs++
        const failures = consecutiveFailuresRef.current.runs
        const errorMessage = error instanceof Error ? error.message : 'Unknown error'

        console.error(`Failed to load runs (attempt ${failures}/${MAX_CONSECUTIVE_FAILURES}):`, error)

        // Check if we've hit max consecutive failures
        if (failures >= MAX_CONSECUTIVE_FAILURES) {
          setIsPollingPaused(true)
          setPollingError(`Failed to connect to server after ${MAX_CONSECUTIVE_FAILURES} attempts. Please check your connection.`)
          console.error('Polling paused due to max consecutive failures. Manual intervention required.')
        } else {
          // Calculate exponential backoff delay
          const backoffDelay = Math.min(
            POLLING_INTERVALS.ACTIVE_RUNS * Math.pow(BACKOFF_MULTIPLIER, failures - 1),
            MAX_BACKOFF_DELAY
          )
          setPollingError(`Connection issue: ${errorMessage}. Retrying...`)

          // Schedule next poll with backoff
          scheduleNextPoll(backoffDelay)
        }
      }
    }

    const scheduleNextPoll = (delay: number) => {
      // Clear any existing timeout/interval
      if (intervalId) clearInterval(intervalId)
      if (timeoutId) clearTimeout(timeoutId)

      // Schedule next poll
      timeoutId = setTimeout(() => {
        void loadRuns()
      }, delay)
    }

    // Start initial poll
    void loadRuns()

    return () => {
      if (intervalId) clearInterval(intervalId)
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [isPollingPaused])

  /**
   * Poll run history with exponential backoff on failures
   */
  useEffect(() => {
    let timeoutId: NodeJS.Timeout | null = null

    const loadHistory = async () => {
      if (isPollingPaused) {
        return
      }

      try {
        const h = await listRunHistory(50)
        setHistory(h)

        // Success: reset failure counter
        consecutiveFailuresRef.current.history = 0

        // Schedule next poll at normal interval
        scheduleNextPoll(POLLING_INTERVALS.RUN_HISTORY)
      } catch (error) {
        consecutiveFailuresRef.current.history++
        const failures = consecutiveFailuresRef.current.history

        console.error(`Failed to load history (attempt ${failures}/${MAX_CONSECUTIVE_FAILURES}):`, error)

        if (failures >= MAX_CONSECUTIVE_FAILURES) {
          // Error already set by runs polling, just stop this poller
          console.error('History polling paused due to max consecutive failures.')
        } else {
          const backoffDelay = Math.min(
            POLLING_INTERVALS.RUN_HISTORY * Math.pow(BACKOFF_MULTIPLIER, failures - 1),
            MAX_BACKOFF_DELAY
          )
          scheduleNextPoll(backoffDelay)
        }
      }
    }

    const scheduleNextPoll = (delay: number) => {
      if (timeoutId) clearTimeout(timeoutId)
      timeoutId = setTimeout(() => {
        void loadHistory()
      }, delay)
    }

    void loadHistory()

    return () => {
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [isPollingPaused])

  /**
   * Poll status of selected run more frequently with exponential backoff on failures
   */
  useEffect(() => {
    if (!selectedRunId) {
      return
    }

    let timeoutId: NodeJS.Timeout | null = null

    const pollStatus = async (runId: string) => {
      if (isPollingPaused) {
        return
      }

      try {
        const status = await getRunStatus(runId)
        setRuns((prev) => ({
          ...prev,
          [status.run_id]: {
            ...(prev[status.run_id] ?? {
              run_id: status.run_id,
              agent_config_name: agentNameRef.current,
              mode: modeRef.current,
            }),
            status: status.status,
            started_at: status.started_at ?? prev[status.run_id]?.started_at ?? null,
            finished_at: status.finished_at ?? prev[status.run_id]?.finished_at ?? null,
            message: status.message,
          },
        }))

        // Success: reset failure counter
        consecutiveFailuresRef.current.selectedRun = 0

        // Schedule next poll at normal interval
        scheduleNextPoll(POLLING_INTERVALS.SELECTED_RUN)
      } catch (error) {
        consecutiveFailuresRef.current.selectedRun++
        const failures = consecutiveFailuresRef.current.selectedRun

        console.error(`Failed to poll run status (attempt ${failures}/${MAX_CONSECUTIVE_FAILURES}):`, error)

        if (failures >= MAX_CONSECUTIVE_FAILURES) {
          console.error('Selected run polling paused due to max consecutive failures.')
        } else {
          const backoffDelay = Math.min(
            POLLING_INTERVALS.SELECTED_RUN * Math.pow(BACKOFF_MULTIPLIER, failures - 1),
            MAX_BACKOFF_DELAY
          )
          scheduleNextPoll(backoffDelay)
        }
      }
    }

    const scheduleNextPoll = (delay: number) => {
      if (timeoutId) clearTimeout(timeoutId)
      timeoutId = setTimeout(() => {
        void pollStatus(selectedRunId)
      }, delay)
    }

    void pollStatus(selectedRunId)

    return () => {
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [selectedRunId, isPollingPaused])

  /**
   * Add a new run to the state
   */
  const addRun = (run: RunRecord) => {
    setRuns((prev) => ({
      ...prev,
      [run.run_id]: run,
    }))
  }

  /**
   * Update an existing run with partial updates
   */
  const updateRun = (runId: string, updates: Partial<RunRecord>) => {
    setRuns((prev) => {
      const existing = prev[runId]
      if (!existing) {
        console.warn(`Cannot update non-existent run: ${runId}`)
        return prev
      }
      return {
        ...prev,
        [runId]: {
          ...existing,
          ...updates,
        },
      }
    })
  }

  return {
    runs,
    history,
    setRuns,
    addRun,
    updateRun,
    pollingError,
    isPollingPaused,
  }
}

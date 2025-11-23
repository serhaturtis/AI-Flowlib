import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { startAgentRun, stopRun, AgentRunRequest } from '../../services/agents'
import type { RunRecord } from './useRunsPolling'

export interface UseAgentRunFormResult {
  // Form state
  agentName: string
  mode: string
  maxCycles: string
  formError: string | null

  // Form setters
  setAgentName: (value: string) => void
  setMode: (value: string) => void
  setMaxCycles: (value: string) => void
  setFormError: (error: string | null) => void

  // Mutations
  startMutation: ReturnType<typeof useMutation<{ run_id: string; status: string }, Error, AgentRunRequest>>
  stopMutation: ReturnType<typeof useMutation<{ run_id: string; status: string; started_at: string | null; finished_at: string | null; message: string | null }, Error, string>>

  // Actions
  handleStart: (projectId: string) => void
}

/**
 * Hook for managing agent run form state and mutations.
 *
 * Features:
 * - Form field state management
 * - Validation
 * - Start/stop mutations with React Query
 * - Error handling
 *
 * @param onRunStarted - Callback when a run is successfully started
 * @param onRunStopped - Callback when a run is successfully stopped
 * @returns Form state and handlers
 */
export function useAgentRunForm(
  onRunStarted: (run: RunRecord) => void,
  onRunStopped: (runId: string, updates: Partial<RunRecord>) => void,
): UseAgentRunFormResult {
  const [agentName, setAgentName] = useState('')
  const [mode, setMode] = useState('autonomous')
  const [maxCycles, setMaxCycles] = useState('10')
  const [formError, setFormError] = useState<string | null>(null)

  const startMutation = useMutation({
    mutationFn: (payload: AgentRunRequest) => startAgentRun(payload),
    onSuccess: (response) => {
      // Notify parent to add run to state
      onRunStarted({
        run_id: response.run_id,
        status: response.status,
        agent_config_name: agentName,
        mode,
        started_at: null,
        finished_at: null,
        message: null,
      })
      setFormError(null)
    },
    onError: (error) => {
      const message = error instanceof Error ? error.message : 'Failed to start run.'
      setFormError(message)
    },
  })

  const stopMutation = useMutation({
    mutationFn: (runId: string) => stopRun(runId),
    onSuccess: (response) => {
      // Notify parent to update run state
      onRunStopped(response.run_id, {
        status: response.status,
        started_at: response.started_at,
        finished_at: response.finished_at,
        message: response.message,
      })
    },
  })

  /**
   * Handle run start with validation
   */
  const handleStart = (projectId: string) => {
    if (!projectId || !agentName) {
      setFormError('Project and agent must be selected.')
      return
    }

    const payload: AgentRunRequest = {
      project_id: projectId,
      agent_config_name: agentName,
      mode,
      execution_config: {},
    }

    if (mode === 'autonomous' && maxCycles) {
      const cycles = Number(maxCycles)
      if (isNaN(cycles) || cycles < 1) {
        setFormError('Max cycles must be a positive number.')
        return
      }
      payload.execution_config = { max_cycles: cycles }
    }

    startMutation.mutate(payload)
  }

  return {
    agentName,
    mode,
    maxCycles,
    formError,
    setAgentName,
    setMode,
    setMaxCycles,
    setFormError,
    startMutation,
    stopMutation,
    handleStart,
  }
}

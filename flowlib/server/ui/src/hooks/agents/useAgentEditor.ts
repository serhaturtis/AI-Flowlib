import { useState, useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { applyAgentStructured, renderAgentContent, diffConfig } from '../../services/configs'

export interface UseAgentEditorResult {
  selected: string
  setSelected: (value: string) => void
  values: Record<string, unknown>
  setValues: (values: Record<string, unknown>) => void
  result: string | null
  error: string | null
  diff: string[] | null
  onFieldChange: (name: string, val: unknown) => void
  computeDiff: () => Promise<void>
  apply: () => Promise<void>
}

/**
 * Hook for managing agent editor state and logic.
 *
 * Features:
 * - Selected agent tracking
 * - Form field values management
 * - Diff computation
 * - Apply changes
 * - Error and success handling
 *
 * @param projectId - Current project ID
 * @returns Editor state and handlers
 */
export function useAgentEditor(projectId: string): UseAgentEditorResult {
  const queryClient = useQueryClient()
  const [selected, setSelected] = useState<string>('')
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [diff, setDiff] = useState<string[] | null>(null)

  // Reset state when selected agent changes
  useEffect(() => {
    setValues({})
    setResult(null)
    setError(null)
    setDiff(null)
  }, [selected])

  const onFieldChange = (name: string, val: unknown) => {
    setValues((prev) => ({ ...prev, [name]: val }))
  }

  const computeDiff = async () => {
    try {
      setError(null)
      setResult(null)
      if (!selected) throw new Error('Select an agent.')
      const content = (
        await renderAgentContent({
          name: selected,
          persona: String(values['persona'] ?? ''),
          allowed_tool_categories: (values['allowed_tool_categories'] as string[]) ?? [],
          description: undefined,
        })
      ).content
      const resp = await diffConfig({
        project_id: projectId,
        relative_path: `agents/${selected}.py`,
        proposed_content: content,
      })
      setDiff(resp.diff)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to compute diff.')
    }
  }

  const apply = async () => {
    try {
      setError(null)
      setResult(null)
      if (!selected) throw new Error('Select an agent.')
      await applyAgentStructured({
        project_id: projectId,
        name: selected,
        persona: String(values['persona'] ?? ''),
        allowed_tool_categories: (values['allowed_tool_categories'] as string[]) ?? [],
        model_name: String(values['model_name'] ?? 'default-model'),
        llm_name: String(values['llm_name'] ?? 'default-llm'),
        temperature: Number(values['temperature'] ?? 0.7),
        max_iterations: Number(values['max_iterations'] ?? 10),
        enable_learning: Boolean(values['enable_learning'] ?? false),
        verbose: Boolean(values['verbose'] ?? false),
      })
      setResult('Applied successfully.')
      queryClient.invalidateQueries({ queryKey: ['agents', projectId] })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to apply.')
    }
  }

  return {
    selected,
    setSelected,
    values,
    setValues,
    result,
    error,
    diff,
    onFieldChange,
    computeDiff,
    apply,
  }
}

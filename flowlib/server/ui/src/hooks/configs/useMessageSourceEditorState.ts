import { useEffect, useState } from 'react'
import { fetchMessageSourceSchema, fetchMessageSourceTypes, SchemaResponse } from '../../services/configs'
import { stringifyJson } from '../../utils/configs/configHelpers'
import type { EditorTarget } from '../../components/configs/ConfigEditor/ConfigEditor'

export interface UseMessageSourceEditorStateResult {
  name: string
  setName: (value: string) => void
  sourceType: string
  setSourceType: (value: string) => void
  enabled: boolean
  setEnabled: (value: boolean) => void
  settingsJson: string
  setSettingsJson: (value: string) => void
  typeOptions: string[]
  schema: SchemaResponse | null
  schemaLoading: boolean
  schemaError: string | null
}

/**
 * Hook for managing message source editor state.
 *
 * Features:
 * - Form field state management
 * - Schema loading based on source type
 * - Source type options loading
 * - Auto-initialization from target
 *
 * @param target - Current editing target
 * @param projectId - Project ID (unused but kept for consistency)
 * @returns Message source editor state and handlers
 */
export function useMessageSourceEditorState(
  target: EditorTarget | null,
  projectId: string,
): UseMessageSourceEditorStateResult {
  // Suppress unused variable warning - kept for API consistency
  void projectId

  const [name, setName] = useState<string>('')
  const [sourceType, setSourceType] = useState<string>('timer')
  const [enabled, setEnabled] = useState<boolean>(true)
  const [settingsJson, setSettingsJson] = useState<string>('{}')
  const [typeOptions, setTypeOptions] = useState<string[]>([])
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [schemaLoading, setSchemaLoading] = useState(false)
  const [schemaError, setSchemaError] = useState<string | null>(null)

  // Initialize state when target changes
  useEffect(() => {
    if (!target || target.type !== 'message_source') {
      return
    }

    setName(target.name)
    setSourceType(target.sourceType || 'timer')

    // Extract enabled from data if present
    const data = target.data as Record<string, unknown> | undefined
    if (data && typeof data.enabled === 'boolean') {
      setEnabled(data.enabled)
    } else {
      setEnabled(true)
    }

    // Settings JSON excludes 'enabled' field (handled separately)
    if (data) {
      const { enabled: _enabled, ...settings } = data
      void _enabled // Mark as intentionally unused
      const result = stringifyJson(settings)
      setSettingsJson(result.data ?? '{}')
    } else {
      setSettingsJson('{}')
    }

    // Load source type options
    fetchMessageSourceTypes()
      .then((opts) => setTypeOptions(opts))
      .catch(() => setTypeOptions([]))
  }, [target])

  // Load schema when source type changes
  // Also reset settings when user changes source type (different from target's original type)
  useEffect(() => {
    if (!target || target.type !== 'message_source') {
      return
    }

    const originalSourceType = target.sourceType || 'timer'

    // If user changed source type to something different from the original,
    // reset settings to empty object. This prevents stale fields from the
    // previous source type being persisted.
    if (sourceType !== originalSourceType) {
      setSettingsJson('{}')
    }

    setSchemaLoading(true)
    setSchemaError(null)

    fetchMessageSourceSchema(sourceType)
      .then((s) => setSchema(s))
      .catch((err) => setSchemaError(err instanceof Error ? err.message : 'Failed to load schema.'))
      .finally(() => setSchemaLoading(false))
  }, [target, sourceType])

  return {
    name,
    setName,
    sourceType,
    setSourceType,
    enabled,
    setEnabled,
    settingsJson,
    setSettingsJson,
    typeOptions,
    schema,
    schemaLoading,
    schemaError,
  }
}

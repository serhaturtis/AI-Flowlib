import { useEffect, useState } from 'react'
import { fetchProviderSchema, listProviderTypes, SchemaResponse } from '../../services/configs'
import { stringifyJson } from '../../utils/configs/configHelpers'
import type { EditorTarget } from '../../components/configs/ConfigEditor/ConfigEditor'

export interface UseProviderEditorStateResult {
  name: string
  setName: (value: string) => void
  resourceType: string
  setResourceType: (value: string) => void
  providerType: string
  setProviderType: (value: string) => void
  settingsJson: string
  setSettingsJson: (value: string) => void
  typeOptions: string[]
  schema: SchemaResponse | null
  schemaLoading: boolean
  schemaError: string | null
}

/**
 * Hook for managing provider editor state.
 *
 * Features:
 * - Form field state management
 * - Schema loading based on resource type and provider type
 * - Provider type options loading
 * - Auto-initialization from target
 *
 * @param target - Current editing target
 * @param projectId - Project ID for schema fetching
 * @returns Provider editor state and handlers
 */
export function useProviderEditorState(
  target: EditorTarget | null,
  projectId: string,
): UseProviderEditorStateResult {
  const [name, setName] = useState<string>('')
  const [resourceType, setResourceType] = useState<string>('llm_config')
  const [providerType, setProviderType] = useState<string>('')
  const [settingsJson, setSettingsJson] = useState<string>('{}')
  const [typeOptions, setTypeOptions] = useState<string[]>([])
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [schemaLoading, setSchemaLoading] = useState(false)
  const [schemaError, setSchemaError] = useState<string | null>(null)

  // Initialize state when target changes
  useEffect(() => {
    if (!target || target.type !== 'provider') {
      return
    }

    setName(target.name)
    setResourceType(target.resourceType || 'llm_config')
    setProviderType(target.providerType || '')
    if (typeof target.data === 'string') {
      setSettingsJson(target.data as string)
    } else {
      const result = stringifyJson(target.data)
      setSettingsJson(result.data ?? '{}')
    }

    // Load provider implementation options for this resource type
    if (target.resourceType) {
      listProviderTypes(target.resourceType)
        .then((opts) => setTypeOptions(opts))
        .catch(() => setTypeOptions([]))
    } else {
      setTypeOptions([])
    }

    // Load schema
    if (target.resourceType) {
      setSchemaLoading(true)
      setSchemaError(null)
      fetchProviderSchema(target.resourceType, projectId, target.providerType || undefined)
        .then((s) => setSchema(s))
        .catch((err) => setSchemaError(err instanceof Error ? err.message : 'Failed to load schema.'))
        .finally(() => setSchemaLoading(false))
    } else {
      setSchema(null)
    }
  }, [target, projectId])

  return {
    name,
    setName,
    resourceType,
    setResourceType,
    providerType,
    setProviderType,
    settingsJson,
    setSettingsJson,
    typeOptions,
    schema,
    schemaLoading,
    schemaError,
  }
}

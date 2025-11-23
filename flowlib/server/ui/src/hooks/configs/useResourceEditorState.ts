import { useEffect, useState } from 'react'
import { fetchResourceSchema, listProviderTypes, SchemaResponse } from '../../services/configs'
import { stringifyJson } from '../../utils/configs/configHelpers'
import type { EditorTarget } from '../../components/configs/ConfigEditor/ConfigEditor'

export interface UseResourceEditorStateResult {
  name: string
  setName: (value: string) => void
  resourceType: string
  setResourceType: (value: string) => void
  providerType: string
  setProviderType: (value: string) => void
  configJson: string
  setConfigJson: (value: string) => void
  typeOptions: string[]
  schema: SchemaResponse | null
  schemaLoading: boolean
  schemaError: string | null
}

/**
 * Hook for managing resource editor state.
 *
 * Features:
 * - Form field state management
 * - Schema loading based on resource type and provider type
 * - Provider type options loading
 * - Auto-initialization from target
 *
 * @param target - Current editing target
 * @param projectId - Project ID for schema fetching
 * @returns Resource editor state and handlers
 */
export function useResourceEditorState(
  target: EditorTarget | null,
  projectId: string,
): UseResourceEditorStateResult {
  const [name, setName] = useState<string>('')
  const [resourceType, setResourceType] = useState<string>('model_config')
  const [providerType, setProviderType] = useState<string>('')
  const [configJson, setConfigJson] = useState<string>('{}')
  const [typeOptions, setTypeOptions] = useState<string[]>([])
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [schemaLoading, setSchemaLoading] = useState(false)
  const [schemaError, setSchemaError] = useState<string | null>(null)

  // Initialize state when target changes
  useEffect(() => {
    if (!target || target.type !== 'resource') {
      return
    }

    setName(target.name)
    setResourceType(target.resourceType || 'model_config')
    setProviderType(target.providerType || '')
    if (typeof target.data === 'string') {
      setConfigJson(target.data as string)
    } else {
      const result = stringifyJson(target.data)
      setConfigJson(result.data ?? '{}')
    }

    // Load provider implementation options for this resource type
    // Only fetch provider types for provider-backed resource types
    const providerBackedTypes = [
      'llm_config',
      'multimodal_llm_config',
      'vector_db_config',
      'database_config',
      'cache_config',
      'storage_config',
      'embedding_config',
      'graph_db_config',
      'message_queue_config',
    ]

    if (target.resourceType && providerBackedTypes.includes(target.resourceType)) {
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
      fetchResourceSchema(target.resourceType, projectId, target.providerType || undefined)
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
    configJson,
    setConfigJson,
    typeOptions,
    schema,
    schemaLoading,
    schemaError,
  }
}

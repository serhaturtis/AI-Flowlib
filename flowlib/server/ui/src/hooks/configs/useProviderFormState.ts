import { useState, useEffect, useCallback } from 'react'
import { fetchProviderSchema, listProviderTypes, SchemaResponse } from '../../services/configs'
import { stringifyJson } from '../../utils/configs/configHelpers'

export interface UseProviderFormStateResult {
  resourceType: string
  setResourceType: (value: string) => void
  providerType: string
  setProviderType: (value: string) => void
  description: string
  setDescription: (value: string) => void
  settingsJson: string
  setSettingsJson: (value: string) => void
  typeOptions: string[]
  schema: SchemaResponse | null
  schemaLoading: boolean
  schemaError: string | null
  resetForm: () => void
}

/**
 * Hook for managing provider creation form state.
 *
 * Features:
 * - Form field state management
 * - Schema loading based on resource type and provider type
 * - Provider type options loading
 * - Default value population from schema
 * - Form reset with schema defaults
 *
 * @param projectId - Project ID for schema fetching
 * @param isActive - Whether this form is currently active
 * @returns Provider form state and handlers
 */
export function useProviderFormState(projectId: string, isActive: boolean): UseProviderFormStateResult {
  const [resourceType, setResourceType] = useState('llm_config')
  const [providerType, setProviderType] = useState('llamacpp')
  const [description, setDescription] = useState('Auto-generated provider config.')
  const [settingsJson, setSettingsJson] = useState('{\n  "config": {}\n}')
  const [typeOptions, setTypeOptions] = useState<string[]>([])
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [schemaLoading, setSchemaLoading] = useState(false)
  const [schemaError, setSchemaError] = useState<string | null>(null)

  /**
   * Reset form to default values (with schema defaults if available)
   */
  const resetForm = useCallback(() => {
    if (schema) {
      const providerTypeField = schema.fields.find((field) => field.name === 'provider_type')
      setProviderType(
        providerTypeField && typeof providerTypeField.default === 'string' ? providerTypeField.default : '',
      )
      const settingsField = schema.fields.find((field) => field.name === 'settings')
      if (settingsField?.default) {
        const result = stringifyJson(settingsField.default)
        setSettingsJson(result.data ?? '{\n  "config": {}\n}')
      } else {
        setSettingsJson('{\n  "config": {}\n}')
      }
    } else {
      setProviderType('')
      setSettingsJson('{\n  "config": {}\n}')
    }
  }, [schema])

  /**
   * Load provider schema based on resource type and provider type
   */
  useEffect(() => {
    if (!isActive) {
      setSchema(null)
      return
    }
    if (!projectId) {
      setSchema(null)
      return
    }

    let cancelled = false
    setSchemaLoading(true)
    setSchemaError(null)

    // Load provider type options
    listProviderTypes(resourceType)
      .then((opts) => {
        if (!cancelled) setTypeOptions(opts)
      })
      .catch(() => {
        if (!cancelled) setTypeOptions([])
      })

    // Load schema
    fetchProviderSchema(resourceType, projectId, providerType || undefined)
      .then((loadedSchema) => {
        if (cancelled) return
        setSchema(loadedSchema)

        // Auto-populate defaults from schema
        const providerTypeField = loadedSchema.fields.find((field) => field.name === 'provider_type')
        if (!providerType && providerTypeField?.default) {
          setProviderType(String(providerTypeField.default))
        }
        const settingsField = loadedSchema.fields.find((field) => field.name === 'settings')
        if (settingsField?.default) {
          const result = stringifyJson(settingsField.default)
          if (result.data) {
            setSettingsJson(result.data)
          }
        }
      })
      .catch((error) => {
        if (cancelled) return
        const message = error instanceof Error ? error.message : 'Failed to load provider schema.'
        setSchemaError(message)
        setSchema(null)
      })
      .finally(() => {
        if (!cancelled) {
          setSchemaLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [resourceType, providerType, isActive, projectId])

  return {
    resourceType,
    setResourceType,
    providerType,
    setProviderType,
    description,
    setDescription,
    settingsJson,
    setSettingsJson,
    typeOptions,
    schema,
    schemaLoading,
    schemaError,
    resetForm,
  }
}

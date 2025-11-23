import { useState, useEffect, useCallback } from 'react'
import { fetchResourceSchema, SchemaResponse } from '../../services/configs'
import { stringifyJson } from '../../utils/configs/configHelpers'

export interface UseResourceFormStateResult {
  resourceType: string
  setResourceType: (value: string) => void
  providerType: string
  setProviderType: (value: string) => void
  description: string
  setDescription: (value: string) => void
  configJson: string
  setConfigJson: (value: string) => void
  schema: SchemaResponse | null
  schemaLoading: boolean
  schemaError: string | null
  resetForm: () => void
}

/**
 * Hook for managing resource creation form state.
 *
 * Features:
 * - Form field state management
 * - Schema loading based on resource type and provider type
 * - Default value population from schema
 * - Form reset with schema defaults
 *
 * @param projectId - Project ID for schema fetching
 * @param isActive - Whether this form is currently active
 * @returns Resource form state and handlers
 */
export function useResourceFormState(projectId: string, isActive: boolean): UseResourceFormStateResult {
  const [resourceType, setResourceType] = useState('model_config')
  const [providerType, setProviderType] = useState('llamacpp')
  const [description, setDescription] = useState('Auto-generated model config.')
  const [configJson, setConfigJson] = useState('{\n  "model_name": ""\n}')
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [schemaLoading, setSchemaLoading] = useState(false)
  const [schemaError, setSchemaError] = useState<string | null>(null)

  /**
   * Reset form to default values (with schema defaults if available)
   */
  const resetForm = useCallback(() => {
    if (schema) {
      const providerField = schema.fields.find((field) => field.name === 'provider_type')
      setProviderType(providerField && typeof providerField.default === 'string' ? providerField.default : '')
      const configField = schema.fields.find((field) => field.name === 'config')
      if (configField?.default) {
        const result = stringifyJson(configField.default)
        setConfigJson(result.data ?? '{\n  "model_name": ""\n}')
      } else {
        setConfigJson('{\n  "model_name": ""\n}')
      }
    } else {
      setProviderType('')
      setConfigJson('{\n  "model_name": ""\n}')
    }
  }, [schema])

  /**
   * Load resource schema based on resource type and provider type
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

    fetchResourceSchema(resourceType, projectId, providerType || undefined)
      .then((loadedSchema) => {
        if (cancelled) return
        setSchema(loadedSchema)

        // Auto-populate defaults from schema
        const providerField = loadedSchema.fields.find((field) => field.name === 'provider_type')
        if (!providerType && providerField?.default) {
          setProviderType(String(providerField.default))
        }
        const configField = loadedSchema.fields.find((field) => field.name === 'config')
        if (configField?.default) {
          const result = stringifyJson(configField.default)
          if (result.data) {
            setConfigJson(result.data)
          }
        }
      })
      .catch((error) => {
        if (cancelled) return
        const message = error instanceof Error ? error.message : 'Failed to load resource schema.'
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
    configJson,
    setConfigJson,
    schema,
    schemaLoading,
    schemaError,
    resetForm,
  }
}

import { useState, useEffect, useCallback } from 'react'
import { fetchMessageSourceSchema, fetchMessageSourceTypes, SchemaResponse } from '../../services/configs'
import { stringifyJson } from '../../utils/configs/configHelpers'

export interface UseMessageSourceFormStateResult {
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
  resetForm: () => void
}

/**
 * Hook for managing message source creation form state.
 *
 * Features:
 * - Form field state management
 * - Schema loading based on source type
 * - Source type options loading
 * - Default value population from schema
 * - Form reset with schema defaults
 *
 * @param isActive - Whether this form is currently active
 * @returns Message source form state and handlers
 */
export function useMessageSourceFormState(isActive: boolean): UseMessageSourceFormStateResult {
  const [sourceType, setSourceType] = useState('timer')
  const [enabled, setEnabled] = useState(true)
  const [settingsJson, setSettingsJson] = useState('{}')
  const [typeOptions, setTypeOptions] = useState<string[]>([])
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [schemaLoading, setSchemaLoading] = useState(false)
  const [schemaError, setSchemaError] = useState<string | null>(null)

  /**
   * Reset form to default values from schema.
   * If schema is not loaded, settings are cleared to empty object.
   */
  const resetForm = useCallback(() => {
    setEnabled(true)
    if (schema) {
      // Build default settings from schema fields
      const defaults: Record<string, unknown> = {}
      for (const field of schema.fields) {
        if (field.name !== 'name' && field.name !== 'enabled' && field.default !== null && field.default !== undefined) {
          defaults[field.name] = field.default
        }
      }
      const result = stringifyJson(defaults)
      setSettingsJson(result.data ?? '{}')
    } else {
      // No schema loaded - clear to empty, user must wait for schema
      setSettingsJson('{}')
    }
  }, [schema])

  /**
   * Load message source schema based on source type
   */
  useEffect(() => {
    if (!isActive) {
      setSchema(null)
      return
    }

    let cancelled = false
    setSchemaLoading(true)
    setSchemaError(null)

    // Load source type options
    fetchMessageSourceTypes()
      .then((opts) => {
        if (!cancelled) setTypeOptions(opts)
      })
      .catch((error) => {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : 'Failed to load message source types.'
          setSchemaError(message)
          setTypeOptions([])
        }
      })

    // Load schema for current source type
    fetchMessageSourceSchema(sourceType)
      .then((loadedSchema) => {
        if (cancelled) return
        setSchema(loadedSchema)

        // Auto-populate defaults from schema
        const defaults: Record<string, unknown> = {}
        for (const field of loadedSchema.fields) {
          if (field.name !== 'name' && field.name !== 'enabled' && field.default !== null && field.default !== undefined) {
            defaults[field.name] = field.default
          }
        }
        const result = stringifyJson(defaults)
        if (result.data) {
          setSettingsJson(result.data)
        }
      })
      .catch((error) => {
        if (cancelled) return
        const message = error instanceof Error ? error.message : 'Failed to load message source schema.'
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
  }, [sourceType, isActive])

  return {
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
    resetForm,
  }
}

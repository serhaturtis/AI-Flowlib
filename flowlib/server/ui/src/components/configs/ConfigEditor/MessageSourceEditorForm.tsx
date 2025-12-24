import { useMemo, useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { SchemaFieldInput } from '../SchemaFields/SchemaFieldInput'
import { RenameDeleteControls } from '../RenameDeleteControls'
import { DiffPreview } from './DiffPreview'
import { Button } from '../../ui/Button'
import { Input } from '../../ui/Input'
import { Select } from '../../ui/Select'
import { Label } from '../../ui/Label'
import { parseJson, stringifyJson } from '../../../utils/configs/configHelpers'
import type { EditorTarget } from './ConfigEditor'
import type { UseMessageSourceEditorStateResult } from '../../../hooks/configs/useMessageSourceEditorState'
import type { UseConfigMutationsResult } from '../../../hooks/configs/useConfigMutations'

export interface MessageSourceEditorFormProps {
  projectId: string
  target: EditorTarget
  messageSourceState: UseMessageSourceEditorStateResult
  mutations: UseConfigMutationsResult
  onSuccess: (message: string) => void
  onError: (message: string) => void
  onDiff: () => void
  onApply: () => void
}

/**
 * Structured editor form for message source configurations.
 * Uses SchemaFieldInput components to render dynamic form fields based on schema.
 */
export function MessageSourceEditorForm({
  projectId,
  target,
  messageSourceState,
  mutations,
  onSuccess,
  onError,
  onDiff,
  onApply,
}: MessageSourceEditorFormProps) {
  const queryClient = useQueryClient()

  // Parse the settings JSON into an object for field binding
  const settingsObject = useMemo(() => {
    const result = parseJson<Record<string, unknown>>(messageSourceState.settingsJson)
    return result.success && result.data ? result.data : {}
  }, [messageSourceState.settingsJson])

  // Handle field value changes - update the settingsJson
  const handleFieldChange = useCallback(
    (fieldName: string, value: unknown) => {
      const newSettings = { ...settingsObject, [fieldName]: value }
      const result = stringifyJson(newSettings)
      if (result.data) {
        messageSourceState.setSettingsJson(result.data)
      }
    },
    [settingsObject, messageSourceState]
  )

  // Get fields to render (exclude name and enabled which are handled separately)
  const fieldsToRender = useMemo(() => {
    if (!messageSourceState.schema) return []
    return messageSourceState.schema.fields.filter(
      (f) => f.name !== 'name' && f.name !== 'enabled'
    )
  }, [messageSourceState.schema])

  return (
    <form className="grid gap-4 max-w-2xl">
      <div className="grid gap-2">
        <Label htmlFor="message-source-name">Name</Label>
        <Input
          id="message-source-name"
          type="text"
          value={messageSourceState.name || target.name}
          onChange={(e) => messageSourceState.setName(e.target.value)}
        />
        {mutations.structuredErrors['message_source.name'] && (
          <span className="text-sm text-destructive">{mutations.structuredErrors['message_source.name']}</span>
        )}
      </div>

      <div className="grid gap-2">
        <Label htmlFor="message-source-type">Source Type</Label>
        {messageSourceState.typeOptions.length > 0 ? (
          <Select
            id="message-source-type"
            value={messageSourceState.sourceType}
            onChange={(e) => messageSourceState.setSourceType(e.target.value)}
          >
            {messageSourceState.typeOptions.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </Select>
        ) : (
          <Input
            id="message-source-type"
            type="text"
            value={messageSourceState.sourceType}
            readOnly
            className="bg-muted cursor-not-allowed"
          />
        )}
        {mutations.structuredErrors['message_source.source_type'] && (
          <span className="text-sm text-destructive">{mutations.structuredErrors['message_source.source_type']}</span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="message-source-enabled"
          checked={messageSourceState.enabled}
          onChange={(e) => messageSourceState.setEnabled(e.target.checked)}
          className="h-4 w-4 rounded border-border"
        />
        <Label htmlFor="message-source-enabled">Enabled</Label>
      </div>

      {messageSourceState.schemaLoading && <p>Loading schema...</p>}
      {messageSourceState.schemaError && <p style={{ color: 'red' }}>{messageSourceState.schemaError}</p>}

      {/* Dynamic schema fields */}
      {fieldsToRender.map((field) => (
        <SchemaFieldInput
          key={`edit-message-source-${field.name}`}
          keyPrefix="edit-message-source"
          field={field}
          value={settingsObject[field.name]}
          onChange={(val) => handleFieldChange(field.name, val)}
        />
      ))}

      {mutations.structuredErrors['message_source.settings'] && (
        <span className="text-sm text-destructive">{mutations.structuredErrors['message_source.settings']}</span>
      )}

      {/* Rename/Delete actions */}
      <fieldset className="border border-border rounded-md p-4">
        <legend className="px-2 text-sm font-medium">File Operations</legend>
        <div className="grid gap-3">
          <div className="grid gap-2">
            <Label htmlFor="current-path">Current Path</Label>
            <Input
              id="current-path"
              type="text"
              value={target.relativePath}
              readOnly
              className="bg-muted cursor-not-allowed"
            />
          </div>
          <RenameDeleteControls
            projectId={projectId}
            currentPath={target.relativePath}
            onRenamed={() => {
              onSuccess('Renamed successfully.')
              queryClient.invalidateQueries({ queryKey: ['configs'] })
            }}
            onDeleted={() => {
              onSuccess('Deleted successfully.')
              queryClient.invalidateQueries({ queryKey: ['configs'] })
            }}
            onError={onError}
          />
        </div>
      </fieldset>

      <div className="flex gap-3">
        <Button type="button" onClick={onDiff} disabled={mutations.computeStructuredDiff.isPending}>
          {mutations.computeStructuredDiff.isPending ? 'Diffing...' : 'Show Diff'}
        </Button>
        <Button type="button" onClick={onApply} disabled={mutations.applyStructuredMutation.isPending} variant="secondary">
          {mutations.applyStructuredMutation.isPending ? 'Applying...' : 'Apply'}
        </Button>
      </div>

      <DiffPreview diffResult={mutations.structuredDiff} />
    </form>
  )
}

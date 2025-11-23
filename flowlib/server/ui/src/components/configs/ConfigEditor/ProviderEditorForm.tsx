import { useQueryClient } from '@tanstack/react-query'
import { SchemaFieldInput } from '../SchemaFields/SchemaFieldInput'
import { RenameDeleteControls } from '../RenameDeleteControls'
import { DiffPreview } from './DiffPreview'
import { Button } from '../../ui/Button'
import { Input } from '../../ui/Input'
import { Textarea } from '../../ui/Textarea'
import { Select } from '../../ui/Select'
import { Label } from '../../ui/Label'
import type { EditorTarget } from './ConfigEditor'
import type { UseProviderEditorStateResult } from '../../../hooks/configs/useProviderEditorState'
import type { UseConfigMutationsResult } from '../../../hooks/configs/useConfigMutations'

export interface ProviderEditorFormProps {
  projectId: string
  target: EditorTarget
  providerState: UseProviderEditorStateResult
  mutations: UseConfigMutationsResult
  onSuccess: (message: string) => void
  onError: (message: string) => void
  onDiff: () => void
  onApply: () => void
}

/**
 * Structured editor form for provider configurations.
 */
export function ProviderEditorForm({
  projectId,
  target,
  providerState,
  mutations,
  onSuccess,
  onError,
  onDiff,
  onApply,
}: ProviderEditorFormProps) {
  const queryClient = useQueryClient()

  return (
    <form className="grid gap-4 max-w-2xl">
      <div className="grid gap-2">
        <Label htmlFor="provider-name">Name</Label>
        <Input
          id="provider-name"
          type="text"
          value={providerState.name || target.name}
          onChange={(e) => providerState.setName(e.target.value)}
        />
        {mutations.structuredErrors['provider.name'] && (
          <span className="text-sm text-destructive">{mutations.structuredErrors['provider.name']}</span>
        )}
      </div>

      <div className="grid gap-2">
        <Label htmlFor="provider-resource-type">Resource Type</Label>
        <Input
          id="provider-resource-type"
          type="text"
          value={providerState.resourceType}
          readOnly
          className="bg-muted cursor-not-allowed"
        />
        {mutations.structuredErrors['provider.resource_type'] && (
          <span className="text-sm text-destructive">{mutations.structuredErrors['provider.resource_type']}</span>
        )}
      </div>

      {providerState.schemaLoading && <p>Loading schema…</p>}
      {providerState.schemaError && <p className="text-sm text-destructive">{providerState.schemaError}</p>}

      {providerState.schema ? (
        <>
          {providerState.schema.fields
            .filter((f) => f.name !== 'name')
            .map((field) => {
              if (field.name === 'provider_type') {
                return (
                  <div key="edit-provider-provider-type" className="grid gap-2">
                    <Label htmlFor="edit-provider-provider-type-input">Provider Type</Label>
                    {providerState.typeOptions.length > 0 ? (
                      <Select
                        id="edit-provider-provider-type-input"
                        value={providerState.providerType}
                        onChange={(e) => providerState.setProviderType(e.target.value)}
                      >
                        <option value="">Select provider type</option>
                        {providerState.typeOptions.map((opt) => (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ))}
                      </Select>
                    ) : (
                      <Input
                        id="edit-provider-provider-type-input"
                        type="text"
                        value={providerState.providerType}
                        onChange={(e) => providerState.setProviderType(e.target.value)}
                      />
                    )}
                  </div>
                )
              }
              if (field.name === 'settings') {
                return (
                  <SchemaFieldInput
                    key="edit-provider-settings"
                    keyPrefix="edit-provider"
                    field={field}
                    value={providerState.settingsJson}
                    onChange={(val) => providerState.setSettingsJson(String(val ?? ''))}
                  />
                )
              }
              return (
                <SchemaFieldInput
                  key={`edit-provider-${field.name}`}
                  keyPrefix="edit-provider"
                  field={field}
                  value=""
                  onChange={() => undefined}
                />
              )
            })}
        </>
      ) : (
        <>
          <div className="grid gap-2">
            <Label htmlFor="provider-provider-type">Provider Type</Label>
            {providerState.typeOptions.length > 0 ? (
              <Select
                id="provider-provider-type"
                value={providerState.providerType}
                onChange={(e) => providerState.setProviderType(e.target.value)}
              >
                <option value="">Select provider type</option>
                {providerState.typeOptions.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </Select>
            ) : (
              <Input
                id="provider-provider-type"
                type="text"
                value={providerState.providerType}
                onChange={(e) => providerState.setProviderType(e.target.value)}
              />
            )}
            {mutations.structuredErrors['provider.provider_type'] && (
              <span className="text-sm text-destructive">{mutations.structuredErrors['provider.provider_type']}</span>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="provider-settings-json">Settings (JSON)</Label>
            <Textarea
              id="provider-settings-json"
              rows={8}
              value={providerState.settingsJson}
              onChange={(e) => providerState.setSettingsJson(e.target.value)}
              className="font-mono"
            />
            {mutations.structuredErrors['provider.settings'] && (
              <span className="text-sm text-destructive">{mutations.structuredErrors['provider.settings']}</span>
            )}
          </div>
        </>
      )}

      {/* Rename/Delete actions */}
      <fieldset className="border border-border rounded-md p-4">
        <legend className="px-2 text-sm font-medium">File Operations</legend>
        <div className="grid gap-3">
          <div className="grid gap-2">
            <Label htmlFor="provider-current-path">Current Path</Label>
            <Input
              id="provider-current-path"
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
          {mutations.computeStructuredDiff.isPending ? 'Diffing…' : 'Show Diff'}
        </Button>
        <Button type="button" onClick={onApply} disabled={mutations.applyStructuredMutation.isPending} variant="secondary">
          {mutations.applyStructuredMutation.isPending ? 'Applying…' : 'Apply'}
        </Button>
      </div>

      <DiffPreview diffResult={mutations.structuredDiff} />
    </form>
  )
}

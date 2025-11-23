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
import type { UseResourceEditorStateResult } from '../../../hooks/configs/useResourceEditorState'
import type { UseConfigMutationsResult } from '../../../hooks/configs/useConfigMutations'

export interface ResourceEditorFormProps {
  projectId: string
  target: EditorTarget
  resourceState: UseResourceEditorStateResult
  mutations: UseConfigMutationsResult
  onSuccess: (message: string) => void
  onError: (message: string) => void
  onDiff: () => void
  onApply: () => void
}

/**
 * Structured editor form for resource configurations.
 */
export function ResourceEditorForm({
  projectId,
  target,
  resourceState,
  mutations,
  onSuccess,
  onError,
  onDiff,
  onApply,
}: ResourceEditorFormProps) {
  const queryClient = useQueryClient()

  return (
    <form className="grid gap-4 max-w-2xl">
      <div className="grid gap-2">
        <Label htmlFor="resource-name">Name</Label>
        <Input
          id="resource-name"
          type="text"
          value={resourceState.name || target.name}
          onChange={(e) => resourceState.setName(e.target.value)}
        />
        {mutations.structuredErrors['resource.name'] && (
          <span className="text-sm text-destructive">{mutations.structuredErrors['resource.name']}</span>
        )}
      </div>

      <div className="grid gap-2">
        <Label htmlFor="resource-type">Resource Type</Label>
        <Input
          id="resource-type"
          type="text"
          value={resourceState.resourceType}
          readOnly
          className="bg-muted cursor-not-allowed"
        />
        {mutations.structuredErrors['resource.resource_type'] && (
          <span className="text-sm text-destructive">{mutations.structuredErrors['resource.resource_type']}</span>
        )}
      </div>

      {resourceState.schemaLoading && <p>Loading schema…</p>}
      {resourceState.schemaError && <p style={{ color: 'red' }}>{resourceState.schemaError}</p>}

      {resourceState.schema ? (
        <>
          {resourceState.schema.fields
            .filter((f) => f.name !== 'name')
            .map((field) => {
              if (field.name === 'provider_type') {
                return (
                  <div key="edit-resource-provider-type" className="grid gap-2">
                    <Label htmlFor="edit-resource-provider-type-input">Provider Type</Label>
                    {resourceState.typeOptions.length > 0 ? (
                      <Select
                        id="edit-resource-provider-type-input"
                        value={resourceState.providerType}
                        onChange={(e) => resourceState.setProviderType(e.target.value)}
                      >
                        <option value="">Select provider type</option>
                        {resourceState.typeOptions.map((opt) => (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ))}
                      </Select>
                    ) : (
                      <Input
                        id="edit-resource-provider-type-input"
                        type="text"
                        value={resourceState.providerType}
                        onChange={(e) => resourceState.setProviderType(e.target.value)}
                      />
                    )}
                  </div>
                )
              }
              if (field.name === 'config') {
                return (
                  <SchemaFieldInput
                    key="edit-resource-config"
                    keyPrefix="edit-resource"
                    field={field}
                    value={resourceState.configJson}
                    onChange={(val) => resourceState.setConfigJson(String(val ?? ''))}
                  />
                )
              }
              return (
                <SchemaFieldInput
                  key={`edit-resource-${field.name}`}
                  keyPrefix="edit-resource"
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
            <Label htmlFor="resource-provider-type">Provider Type</Label>
            {resourceState.typeOptions.length > 0 ? (
              <Select
                id="resource-provider-type"
                value={resourceState.providerType}
                onChange={(e) => resourceState.setProviderType(e.target.value)}
              >
                <option value="">Select provider type</option>
                {resourceState.typeOptions.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </Select>
            ) : (
              <Input
                id="resource-provider-type"
                type="text"
                value={resourceState.providerType}
                onChange={(e) => resourceState.setProviderType(e.target.value)}
              />
            )}
            {mutations.structuredErrors['resource.provider_type'] && (
              <span className="text-sm text-destructive">{mutations.structuredErrors['resource.provider_type']}</span>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="resource-config-json">Config (JSON)</Label>
            <Textarea
              id="resource-config-json"
              rows={8}
              value={resourceState.configJson}
              onChange={(e) => resourceState.setConfigJson(e.target.value)}
              className="font-mono"
            />
            {mutations.structuredErrors['resource.config'] && (
              <span className="text-sm text-destructive">{mutations.structuredErrors['resource.config']}</span>
            )}
          </div>
        </>
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

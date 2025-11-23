import { AlertCircle } from 'lucide-react'
import { FormField } from '../../forms/FormField'
import { Input } from '../../ui/Input'
import { Select } from '../../ui/Select'
import { Textarea } from '../../ui/Textarea'
import { Alert, AlertDescription } from '../../ui/Alert'
import { Spinner } from '../../ui/Spinner'
import { SchemaFieldInput } from '../SchemaFields/SchemaFieldInput'
import type { UseResourceFormStateResult } from '../../../hooks/configs/useResourceFormState'

export interface ResourceFormFieldsProps {
  resourceState: UseResourceFormStateResult
}

/**
 * Resource-specific form fields for config creation.
 */
export function ResourceFormFields({ resourceState }: ResourceFormFieldsProps) {
  return (
    <>
      <FormField label="Resource Type" required>
        <Select value={resourceState.resourceType} onChange={(event) => resourceState.setResourceType(event.target.value)}>
          <option value="model_config">model_config</option>
        </Select>
      </FormField>

      <FormField label="Description">
        <Input
          type="text"
          value={resourceState.description}
          onChange={(event) => resourceState.setDescription(event.target.value)}
        />
      </FormField>

      {resourceState.schemaLoading && (
        <div className="flex items-center gap-2">
          <Spinner size="sm" />
          <span className="text-sm text-muted-foreground">Loading schema…</span>
        </div>
      )}

      {resourceState.schemaError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{resourceState.schemaError}</AlertDescription>
        </Alert>
      )}

      {resourceState.schema && (
        <div className="space-y-4">
          {resourceState.schema.fields
            .filter((f) => f.name !== 'name')
            .map((field) => {
              if (field.name === 'provider_type') {
                return (
                  <FormField key="resource-provider-type" label="Provider Type" required>
                    <Input
                      type="text"
                      value={resourceState.providerType}
                      onChange={(e) => resourceState.setProviderType(e.target.value)}
                      placeholder="llamacpp"
                    />
                  </FormField>
                )
              }
              if (field.name === 'config') {
                return (
                  <SchemaFieldInput
                    key="resource-config"
                    keyPrefix="create-resource"
                    field={field}
                    value={resourceState.configJson}
                    onChange={(val) => resourceState.setConfigJson(String(val ?? ''))}
                  />
                )
              }
              return (
                <SchemaFieldInput
                  key={`create-resource-${field.name}`}
                  keyPrefix="create-resource"
                  field={field}
                  value=""
                  onChange={() => undefined}
                />
              )
            })}
        </div>
      )}

      {!resourceState.schemaLoading && !resourceState.schema && !resourceState.schemaError && (
        <>
          <FormField label="Provider Type" required>
            <Input
              type="text"
              value={resourceState.providerType}
              onChange={(event) => resourceState.setProviderType(event.target.value)}
              placeholder="llamacpp"
            />
          </FormField>
          <FormField label="Config (JSON)">
            <Textarea
              rows={6}
              value={resourceState.configJson}
              onChange={(event) => {
                resourceState.setConfigJson(event.target.value)
              }}
              className="font-mono"
            />
          </FormField>
        </>
      )}
    </>
  )
}

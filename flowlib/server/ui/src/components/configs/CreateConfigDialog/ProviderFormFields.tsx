import { AlertCircle } from 'lucide-react'
import { FormField } from '../../forms/FormField'
import { Input } from '../../ui/Input'
import { Select } from '../../ui/Select'
import { Textarea } from '../../ui/Textarea'
import { Alert, AlertDescription } from '../../ui/Alert'
import { Spinner } from '../../ui/Spinner'
import { SchemaFieldInput } from '../SchemaFields/SchemaFieldInput'
import type { UseProviderFormStateResult } from '../../../hooks/configs/useProviderFormState'

export interface ProviderFormFieldsProps {
  providerState: UseProviderFormStateResult
}

/**
 * Provider-specific form fields for config creation.
 */
export function ProviderFormFields({ providerState }: ProviderFormFieldsProps) {
  return (
    <>
      <FormField label="Provider Resource Type" required>
        <Select value={providerState.resourceType} onChange={(event) => providerState.setResourceType(event.target.value)}>
          <option value="llm_config">llm_config</option>
          <option value="multimodal_llm_config">multimodal_llm_config</option>
          <option value="vector_db_config">vector_db_config</option>
          <option value="database_config">database_config</option>
          <option value="cache_config">cache_config</option>
          <option value="storage_config">storage_config</option>
          <option value="embedding_config">embedding_config</option>
          <option value="graph_db_config">graph_db_config</option>
          <option value="message_queue_config">message_queue_config</option>
        </Select>
      </FormField>

      <FormField label="Description">
        <Input
          type="text"
          value={providerState.description}
          onChange={(event) => providerState.setDescription(event.target.value)}
        />
      </FormField>

      {providerState.schemaLoading && (
        <div className="flex items-center gap-2">
          <Spinner size="sm" />
          <span className="text-sm text-muted-foreground">Loading schema…</span>
        </div>
      )}

      {providerState.schemaError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{providerState.schemaError}</AlertDescription>
        </Alert>
      )}

      {providerState.schema && (
        <div className="space-y-4">
          {providerState.schema.fields
            .filter((f) => f.name !== 'name')
            .map((field) => {
              if (field.name === 'provider_type') {
                return (
                  <FormField key="provider-type" label="Provider Implementation" required>
                    {providerState.typeOptions.length > 0 ? (
                      <Select
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
                        type="text"
                        value={providerState.providerType}
                        onChange={(e) => providerState.setProviderType(e.target.value)}
                        placeholder="llamacpp"
                      />
                    )}
                  </FormField>
                )
              }
              if (field.name === 'settings') {
                return (
                  <SchemaFieldInput
                    key="provider-settings"
                    keyPrefix="create-provider"
                    field={field}
                    value={providerState.settingsJson}
                    onChange={(val) => providerState.setSettingsJson(String(val ?? ''))}
                  />
                )
              }
              return (
                <SchemaFieldInput
                  key={`create-provider-${field.name}`}
                  keyPrefix="create-provider"
                  field={field}
                  value=""
                  onChange={() => undefined}
                />
              )
            })}
        </div>
      )}

      {!providerState.schemaLoading && !providerState.schema && !providerState.schemaError && (
        <>
          <FormField label="Provider Implementation" required>
            {providerState.typeOptions.length > 0 ? (
              <Select value={providerState.providerType} onChange={(event) => providerState.setProviderType(event.target.value)}>
                <option value="">Select provider type</option>
                {providerState.typeOptions.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </Select>
            ) : (
              <Input
                type="text"
                value={providerState.providerType}
                onChange={(event) => providerState.setProviderType(event.target.value)}
                placeholder="llamacpp"
              />
            )}
          </FormField>
          <FormField label="Settings (JSON)">
            <Textarea
              rows={6}
              value={providerState.settingsJson}
              onChange={(event) => {
                providerState.setSettingsJson(event.target.value)
              }}
              className="font-mono"
            />
          </FormField>
        </>
      )}
    </>
  )
}

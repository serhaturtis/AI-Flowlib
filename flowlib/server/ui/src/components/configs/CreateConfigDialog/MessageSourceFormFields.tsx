import { AlertCircle } from 'lucide-react'
import { FormField } from '../../forms/FormField'
import { Select } from '../../ui/Select'
import { Textarea } from '../../ui/Textarea'
import { Alert, AlertDescription } from '../../ui/Alert'
import { Spinner } from '../../ui/Spinner'
import type { UseMessageSourceFormStateResult } from '../../../hooks/configs/useMessageSourceFormState'

export interface MessageSourceFormFieldsProps {
  messageSourceState: UseMessageSourceFormStateResult
}

/**
 * Message source-specific form fields for config creation.
 */
export function MessageSourceFormFields({ messageSourceState }: MessageSourceFormFieldsProps) {
  return (
    <>
      <FormField label="Source Type" required>
        <Select
          value={messageSourceState.sourceType}
          onChange={(event) => messageSourceState.setSourceType(event.target.value)}
          disabled={messageSourceState.typeOptions.length === 0}
        >
          {messageSourceState.typeOptions.length > 0 ? (
            messageSourceState.typeOptions.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))
          ) : (
            <option value="">Loading source types...</option>
          )}
        </Select>
      </FormField>

      <FormField label="Enabled">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={messageSourceState.enabled}
            onChange={(e) => messageSourceState.setEnabled(e.target.checked)}
            className="h-4 w-4 rounded border-gray-300"
          />
          <span className="text-sm text-muted-foreground">Enable this message source</span>
        </label>
      </FormField>

      {messageSourceState.schemaLoading && (
        <div className="flex items-center gap-2">
          <Spinner size="sm" />
          <span className="text-sm text-muted-foreground">Loading schema…</span>
        </div>
      )}

      {messageSourceState.schemaError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{messageSourceState.schemaError}</AlertDescription>
        </Alert>
      )}

      <FormField label="Settings (JSON)" required>
        <Textarea
          rows={8}
          value={messageSourceState.settingsJson}
          onChange={(event) => {
            messageSourceState.setSettingsJson(event.target.value)
          }}
          className="font-mono text-sm"
          placeholder={getPlaceholder(messageSourceState.sourceType)}
        />
        <p className="text-xs text-muted-foreground mt-1">
          {getHelpText(messageSourceState.sourceType)}
        </p>
      </FormField>
    </>
  )
}

function getPlaceholder(sourceType: string): string {
  switch (sourceType) {
    case 'timer':
      return '{\n  "interval_seconds": 3600,\n  "run_on_start": true,\n  "message_content": "Timer triggered"\n}'
    case 'email':
      return '{\n  "email_provider_name": "my-email-provider",\n  "check_interval_seconds": 300,\n  "folder": "INBOX",\n  "only_unread": true,\n  "mark_as_read": true\n}'
    case 'webhook':
      return '{\n  "path": "/webhook/my-hook",\n  "methods": ["POST"]\n}'
    case 'queue':
      return '{\n  "queue_provider_name": "my-queue",\n  "queue_name": "agent-tasks"\n}'
    default:
      return '{}'
  }
}

function getHelpText(sourceType: string): string {
  switch (sourceType) {
    case 'timer':
      return 'Timer sources trigger at regular intervals. Set interval_seconds for frequency.'
    case 'email':
      return 'Email sources poll an inbox. Requires an email provider config reference.'
    case 'webhook':
      return 'Webhook sources receive HTTP requests. Define path and allowed methods.'
    case 'queue':
      return 'Queue sources consume messages from a queue. Requires a queue provider reference.'
    default:
      return 'Configure source-specific settings as JSON.'
  }
}

import { Code2, Eye, AlertCircle, CheckCircle2 } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Textarea } from '../ui/Textarea'
import { Select } from '../ui/Select'
import { FormField } from '../forms/FormField'
import { Alert, AlertDescription } from '../ui/Alert'
import { Stack } from '../layout/Stack'
import type { UseAgentEditorResult } from '../../hooks/agents/useAgentEditor'
import type { SchemaResponse, SchemaField } from '../../services/configs'

export interface AgentEditorProps {
  editorHook: UseAgentEditorResult
  schema: SchemaResponse
  agents: string[]
}

/**
 * Schema-driven agent editor component.
 */
export function AgentEditor({ editorHook, schema, agents }: AgentEditorProps) {
  const { selected, setSelected, values, onFieldChange, computeDiff, apply, diff, error, result } = editorHook

  const renderField = (field: SchemaField) => {
    const key = field.name
    const v = values[key]
    const description = field.description ?? undefined

    if (field.type === 'string') {
      const str = typeof v === 'string' ? v : v ?? ''
      return (
        <FormField key={key} label={field.name} required={field.required} description={description}>
          <Input type="text" value={str as string} onChange={(e) => onFieldChange(key, e.target.value)} />
        </FormField>
      )
    }
    if (field.type === 'boolean') {
      const boolValue = typeof v === 'boolean' ? v : false
      return (
        <FormField key={key} label={field.name} required={field.required} description={description}>
          <Select value={boolValue ? 'true' : 'false'} onChange={(e) => onFieldChange(key, e.target.value === 'true')}>
            <option value="true">true</option>
            <option value="false">false</option>
          </Select>
        </FormField>
      )
    }
    if (field.type === 'number' || field.type === 'integer') {
      const str = typeof v === 'number' || typeof v === 'string' ? String(v ?? '') : v ?? ''
      return (
        <FormField key={key} label={field.name} required={field.required} description={description}>
          <Input
            type="number"
            value={str as string}
            onChange={(e) => onFieldChange(key, e.target.value)}
            min={field.numeric_min ?? undefined}
            max={field.numeric_max ?? undefined}
          />
        </FormField>
      )
    }
    if (field.type === 'array') {
      const list: string[] =
        typeof v === 'string'
          ? (v as string).split('\n').map((s) => s.trim()).filter(Boolean)
          : Array.isArray(v)
            ? (v as unknown[]).map((x) => String(x ?? '')).filter(Boolean)
            : []
      return (
        <FormField key={key} label={field.name} required={field.required} description={description}>
          <Textarea
            rows={3}
            value={list.join('\n')}
            onChange={(e) =>
              onFieldChange(
                key,
                e.target.value
                  .split('\n')
                  .map((s) => s.trim())
                  .filter(Boolean),
              )
            }
            placeholder="One value per line"
            className="font-mono"
          />
        </FormField>
      )
    }
    return null
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Code2 className="h-5 w-5" />
          Edit Agent (Structured)
        </CardTitle>
        <CardDescription>Edit agent configuration using schema-driven forms</CardDescription>
      </CardHeader>
      <CardContent>
        <Stack spacing="lg" className="max-w-2xl">
          <FormField label="Agent">
            <Select value={selected} onChange={(e) => setSelected(e.target.value)}>
              <option value="">Select agent</option>
              {agents.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </Select>
          </FormField>

          {schema.fields && selected && <div className="space-y-4">{schema.fields.map((f) => renderField(f))}</div>}

          {selected && (
            <div className="flex gap-2">
              <Button type="button" onClick={computeDiff} variant="outline">
                <Eye className="h-4 w-4 mr-2" />
                Show Diff
              </Button>
              <Button type="button" onClick={apply}>
                Apply
              </Button>
            </div>
          )}

          {diff && diff.length > 0 && (
            <div className="bg-[#272822] text-[#f8f8f2] p-4 rounded-md overflow-x-auto">
              <pre className="text-sm">{diff.join('')}</pre>
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {result && (
            <Alert variant="success">
              <CheckCircle2 className="h-4 w-4" />
              <AlertDescription>{result}</AlertDescription>
            </Alert>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

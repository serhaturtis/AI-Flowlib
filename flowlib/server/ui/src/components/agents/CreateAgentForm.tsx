import { Plus, AlertCircle, CheckCircle2 } from 'lucide-react'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Textarea } from '../ui/Textarea'
import { FormField } from '../forms/FormField'
import { Alert, AlertDescription } from '../ui/Alert'
import { Spinner } from '../ui/Spinner'
import { Stack } from '../layout/Stack'
import { KnowledgePluginMultiSelect } from '../knowledge/KnowledgePluginMultiSelect'
import type { UseAgentFormResult } from '../../hooks/agents/useAgentForm'

export interface CreateAgentFormProps {
  formHook: UseAgentFormResult
  projectId: string
}

/**
 * Form for creating new agents.
 */
export function CreateAgentForm({ formHook, projectId }: CreateAgentFormProps) {
  const {
    name,
    setName,
    persona,
    setPersona,
    categoriesText,
    setCategoriesText,
    knowledgePlugins,
    setKnowledgePlugins,
    formError,
    formSuccess,
    isPending,
    handleCreate,
  } = formHook

  return (
    <Stack spacing="lg">
      <FormField label="Name" required>
        <Input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="example-agent" />
      </FormField>

      <FormField label="Persona" required description="Agent's system prompt and personality">
        <Textarea rows={4} value={persona} onChange={(e) => setPersona(e.target.value)} />
      </FormField>

      <FormField
        label="Allowed Categories"
        required
        description="Comma-separated list of tool categories (e.g., generic, programming)"
      >
        <Input
          type="text"
          value={categoriesText}
          onChange={(e) => setCategoriesText(e.target.value)}
          placeholder="generic, programming"
        />
      </FormField>

      <KnowledgePluginMultiSelect
        projectId={projectId}
        selectedPluginIds={knowledgePlugins}
        onChange={setKnowledgePlugins}
      />

      {formError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{formError}</AlertDescription>
        </Alert>
      )}
      {formSuccess && (
        <Alert variant="success">
          <CheckCircle2 className="h-4 w-4" />
          <AlertDescription>{formSuccess}</AlertDescription>
        </Alert>
      )}

      <Button type="button" onClick={handleCreate} disabled={isPending || !projectId} className="w-full">
        {isPending ? (
          <>
            <Spinner size="sm" className="mr-2" />
            Creating…
          </>
        ) : (
          <>
            <Plus className="h-4 w-4 mr-2" />
            Create Agent
          </>
        )}
      </Button>
    </Stack>
  )
}

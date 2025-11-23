import { Plus } from 'lucide-react'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Textarea } from '../ui/Textarea'
import { FormField } from '../forms/FormField'
import { Spinner } from '../ui/Spinner'
import { Stack } from '../layout/Stack'
import type { UseProjectFormResult } from '../../hooks/projects/useProjectForm'

export interface CreateProjectFormProps {
  formHook: UseProjectFormResult
}

/**
 * Form for creating new projects.
 */
export function CreateProjectForm({ formHook }: CreateProjectFormProps) {
  const { form, fieldErrors, formError, isPending, handleChange, handleBlur, handleSubmit } = formHook

  return (
    <form onSubmit={handleSubmit}>
      <Stack spacing="lg">
        <FormField label="Project Name" required error={fieldErrors.name}>
          <Input
            type="text"
            value={form.name}
            onChange={handleChange('name')}
            onBlur={handleBlur('name')}
            placeholder="e.g., research-assistant"
            required
          />
        </FormField>

        <FormField label="Description" description="Short project summary" error={fieldErrors.description}>
          <Textarea
            value={form.description}
            onChange={handleChange('description')}
            onBlur={handleBlur('description')}
            rows={3}
            placeholder="Short project summary"
          />
        </FormField>

        <FormField label="Agent Names" description="Comma-separated list of agent names" error={fieldErrors.agentNames}>
          <Input
            type="text"
            value={form.agentNames}
            onChange={handleChange('agentNames')}
            onBlur={handleBlur('agentNames')}
            placeholder="writer, reviewer"
          />
        </FormField>

        <FormField label="Tool Categories" description="Comma-separated list of tool categories" error={fieldErrors.toolCategories}>
          <Input
            type="text"
            value={form.toolCategories}
            onChange={handleChange('toolCategories')}
            onBlur={handleBlur('toolCategories')}
            placeholder="generic, programming"
          />
        </FormField>

        {formError && (
          <div className="text-sm text-red-600 dark:text-red-400" role="alert">
            {formError}
          </div>
        )}

        <Button type="submit" disabled={isPending} className="w-full">
          {isPending ? (
            <>
              <Spinner size="sm" className="mr-2" />
              Creating…
            </>
          ) : (
            <>
              <Plus className="h-4 w-4 mr-2" />
              Create Project
            </>
          )}
        </Button>
      </Stack>
    </form>
  )
}

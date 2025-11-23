import { FormEvent, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { createProject, ProjectCreateRequest } from '../../services/projects'
import {
  validateProjectName,
  validateDescription,
  validateCommaSeparatedList,
} from '../../utils/validation/validators'

type FormState = {
  name: string
  description: string
  agentNames: string
  toolCategories: string
}

type FieldErrors = {
  [K in keyof FormState]?: string
}

const initialForm: FormState = {
  name: '',
  description: '',
  agentNames: '',
  toolCategories: '',
}

const parseList = (value: string): string[] =>
  value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

export interface UseProjectFormResult {
  form: FormState
  fieldErrors: FieldErrors
  formError: string | null
  isPending: boolean
  handleChange: (field: keyof FormState) => (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => void
  handleBlur: (field: keyof FormState) => () => void
  handleSubmit: (event: FormEvent<HTMLFormElement>) => Promise<void>
  setFormError: (error: string | null) => void
}

/**
 * Hook for managing project creation form state and submission.
 *
 * Features:
 * - Form state management
 * - Form validation
 * - Project creation mutation
 * - Error handling
 * - Auto-reset on success
 *
 * @returns Form state and handlers
 */
export function useProjectForm(): UseProjectFormResult {
  const queryClient = useQueryClient()
  const [form, setForm] = useState<FormState>(initialForm)
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [formError, setFormError] = useState<string | null>(null)

  const { mutateAsync, isPending } = useMutation({
    mutationFn: (payload: ProjectCreateRequest) => createProject(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      setForm(initialForm)
      setFieldErrors({})
      setFormError(null)
    },
  })

  /**
   * Validate a specific field
   */
  const validateField = (field: keyof FormState, value: string): string | undefined => {
    let result
    switch (field) {
      case 'name':
        result = validateProjectName(value)
        break
      case 'description':
        result = validateDescription(value)
        break
      case 'agentNames':
        result = validateCommaSeparatedList(value)
        break
      case 'toolCategories':
        result = validateCommaSeparatedList(value)
        break
      default:
        return undefined
    }
    return result.isValid ? undefined : result.error
  }

  /**
   * Validate all fields
   */
  const validateAllFields = (): boolean => {
    const errors: FieldErrors = {}
    let isValid = true

    for (const field of Object.keys(form) as Array<keyof FormState>) {
      const error = validateField(field, form[field])
      if (error) {
        errors[field] = error
        isValid = false
      }
    }

    setFieldErrors(errors)
    return isValid
  }

  const handleChange =
    (field: keyof FormState) => (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const value = event.target.value
      setForm((prev) => ({ ...prev, [field]: value }))

      // Clear field error on change
      if (fieldErrors[field]) {
        setFieldErrors((prev) => {
          const next = { ...prev }
          delete next[field]
          return next
        })
      }
    }

  const handleBlur = (field: keyof FormState) => () => {
    const error = validateField(field, form[field])
    if (error) {
      setFieldErrors((prev) => ({ ...prev, [field]: error }))
    }
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setFormError(null)

    // Validate all fields before submission
    if (!validateAllFields()) {
      setFormError('Please fix the errors above before submitting.')
      return
    }

    const payload: ProjectCreateRequest = {
      name: form.name.trim(),
      description: form.description.trim(),
      agent_names: parseList(form.agentNames),
      tool_categories: parseList(form.toolCategories),
    }

    try {
      await mutateAsync(payload)
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : 'Failed to create project.'
      setFormError(message)
    }
  }

  return {
    form,
    fieldErrors,
    formError,
    isPending,
    handleChange,
    handleBlur,
    handleSubmit,
    setFormError,
  }
}

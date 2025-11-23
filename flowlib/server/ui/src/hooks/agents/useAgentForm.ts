import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { createAgentConfig } from '../../services/configs'

export interface UseAgentFormResult {
  name: string
  setName: (value: string) => void
  persona: string
  setPersona: (value: string) => void
  categoriesText: string
  setCategoriesText: (value: string) => void
  knowledgePlugins: string[]
  setKnowledgePlugins: (value: string[]) => void
  formError: string | null
  formSuccess: string | null
  isPending: boolean
  handleCreate: () => void
  resetForm: () => void
}

/**
 * Hook for managing agent creation form state and mutation.
 *
 * Features:
 * - Form state management
 * - Agent creation mutation
 * - Validation
 * - Auto-reset on success
 * - Error and success notifications
 *
 * @param projectId - Current project ID
 * @returns Form state and handlers
 */
export function useAgentForm(projectId: string): UseAgentFormResult {
  const queryClient = useQueryClient()
  const [formError, setFormError] = useState<string | null>(null)
  const [formSuccess, setFormSuccess] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [persona, setPersona] = useState('I am a helpful Flowlib agent.')
  const [categoriesText, setCategoriesText] = useState('generic')
  const [knowledgePlugins, setKnowledgePlugins] = useState<string[]>([])

  const createMutation = useMutation({
    mutationFn: async () => {
      const categories = categoriesText
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
      if (!projectId || !name.trim() || !persona.trim() || categories.length === 0) {
        throw new Error('All fields are required.')
      }
      return await createAgentConfig({
        project_id: projectId,
        name: name.trim(),
        persona: persona.trim(),
        allowed_tool_categories: categories,
        knowledge_plugins: knowledgePlugins,
      })
    },
    onSuccess: () => {
      setFormSuccess('Agent created successfully.')
      setFormError(null)
      resetForm()
      queryClient.invalidateQueries({ queryKey: ['agents', projectId] })
    },
    onError: (error) => {
      const message = error instanceof Error ? error.message : 'Failed to create agent.'
      setFormError(message)
      setFormSuccess(null)
    },
  })

  const resetForm = () => {
    setName('')
    setPersona('I am a helpful Flowlib agent.')
    setCategoriesText('generic')
    setKnowledgePlugins([])
  }

  const handleCreate = () => {
    createMutation.mutate()
  }

  return {
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
    isPending: createMutation.isPending,
    handleCreate,
    resetForm,
  }
}

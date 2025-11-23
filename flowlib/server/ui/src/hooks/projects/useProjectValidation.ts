import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { validateProject, ProjectValidationResponse } from '../../services/projects'

export interface UseProjectValidationResult {
  validationResult: {
    projectId: string
    data: ProjectValidationResponse
  } | null
  validationError: string | null
  isValidating: boolean
  validatingProjectId: string | undefined
  handleValidateProject: (projectId: string) => void
  setValidationResult: (result: { projectId: string; data: ProjectValidationResponse } | null) => void
  setValidationError: (error: string | null) => void
}

/**
 * Hook for managing project validation state and logic.
 *
 * Features:
 * - Validation state management
 * - Validation trigger
 * - Error handling
 * - Loading states
 *
 * @returns Validation state and handlers
 */
export function useProjectValidation(): UseProjectValidationResult {
  const [validationResult, setValidationResult] = useState<{
    projectId: string
    data: ProjectValidationResponse
  } | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)

  const {
    mutate: triggerValidation,
    isPending: isValidating,
    variables: validatingProjectId,
  } = useMutation({
    mutationFn: (projectId: string) => validateProject(projectId),
    onSuccess: (result, projectId) => {
      setValidationResult({ projectId, data: result })
      setValidationError(null)
    },
    onError: (validationErr: unknown) => {
      const message = validationErr instanceof Error ? validationErr.message : 'Validation failed.'
      setValidationError(message)
      setValidationResult(null)
    },
  })

  const handleValidateProject = (projectId: string) => {
    setValidationError(null)
    setValidationResult(null)
    triggerValidation(projectId)
  }

  return {
    validationResult,
    validationError,
    isValidating,
    validatingProjectId,
    handleValidateProject,
    setValidationResult,
    setValidationError,
  }
}

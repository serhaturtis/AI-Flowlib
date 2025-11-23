import { useState } from 'react'
import { useMutation, useQueryClient, UseMutationResult } from '@tanstack/react-query'
import {
  diffConfig,
  applyConfig,
  applyProviderStructured,
  applyResourceStructured,
  renderProviderContent,
  renderResourceContent,
  ConfigDiffResponse,
  ConfigApplyResponse,
} from '../../services/configs'
import type { EditorTarget } from '../../components/configs/ConfigEditor/ConfigEditor'

// Type guard for Axios-like error responses
interface AxiosErrorResponse {
  response?: {
    status: number
    data?: {
      detail?: {
        message?: string
        issues?: Array<{ path: string; message: string }>
      }
    }
  }
}

function isAxiosError(error: unknown): error is AxiosErrorResponse {
  return (
    typeof error === 'object' &&
    error !== null &&
    'response' in error &&
    typeof (error as AxiosErrorResponse).response === 'object'
  )
}

interface DiffMutationParams {
  projectId: string
  target: EditorTarget
  content: string
}

interface ApplyMutationParams {
  projectId: string
  target: EditorTarget
  content: string
  baseHash: string
}

interface StructuredMutationParams {
  projectId: string
  target: EditorTarget
  providerName: string
  providerResourceType: string
  providerType: string
  providerSettingsJson: string
  resourceName: string
  resourceType: string
  resourceProviderType: string
  resourceConfigJson: string
}

export interface UseConfigMutationsResult {
  // Raw mode mutations
  diffMutation: UseMutationResult<ConfigDiffResponse, Error, DiffMutationParams>
  applyMutation: UseMutationResult<ConfigApplyResponse, Error, ApplyMutationParams>

  // Structured mode state and mutations
  structuredErrors: Record<string, string>
  setStructuredErrors: (errors: Record<string, string>) => void
  structuredDiff: ConfigDiffResponse | null
  setStructuredDiff: (diff: ConfigDiffResponse | null) => void
  computeStructuredDiff: UseMutationResult<ConfigDiffResponse, Error, StructuredMutationParams>
  applyStructuredMutation: UseMutationResult<ConfigApplyResponse, Error, StructuredMutationParams>
  validateStructured: (params: {
    target: EditorTarget | null
    providerName: string
    providerResourceType: string
    providerType: string
    providerSettingsJson: string
    resourceName: string
    resourceType: string
    resourceProviderType: string
    resourceConfigJson: string
  }) => boolean
}

/**
 * Hook for managing config editor mutations.
 *
 * Features:
 * - Raw mode diff/apply mutations
 * - Structured mode diff/apply mutations
 * - Validation logic for structured mode
 * - Error handling and state management
 *
 * @param onSuccess - Callback when mutations succeed
 * @param onError - Callback when mutations fail
 * @param setDiffResult - Setter for raw mode diff result
 * @param setBaseHash - Setter for base hash after successful apply
 * @returns Mutation handlers and state
 */
export function useConfigMutations(
  onSuccess: (message: string) => void,
  onError: (message: string) => void,
  setDiffResult: (diff: ConfigDiffResponse | null) => void,
  setBaseHash: (hash: string) => void,
): UseConfigMutationsResult {
  const queryClient = useQueryClient()
  const [structuredErrors, setStructuredErrors] = useState<Record<string, string>>({})
  const [structuredDiff, setStructuredDiff] = useState<ConfigDiffResponse | null>(null)

  // Raw mode diff mutation
  const diffMutation = useMutation<ConfigDiffResponse, Error, DiffMutationParams>({
    mutationFn: async ({ projectId, target, content }) => {
      return await diffConfig({
        project_id: projectId,
        relative_path: target.relativePath,
        proposed_content: content,
      })
    },
    onSuccess: (result) => {
      setDiffResult(result)
      onError('') // Clear error
    },
    onError: (error) => {
      const message = error instanceof Error ? error.message : 'Failed to compute diff.'
      onError(message)
      setDiffResult(null)
    },
  })

  // Raw mode apply mutation
  const applyMutation = useMutation<ConfigApplyResponse, Error, ApplyMutationParams>({
    mutationFn: async ({ projectId, target, content, baseHash }) => {
      return await applyConfig({
        project_id: projectId,
        relative_path: target.relativePath,
        content,
        sha256_before: baseHash || '',
      })
    },
    onSuccess: (result) => {
      onSuccess('Config applied successfully.')
      setDiffResult(null)
      setBaseHash(result.sha256_after)
      queryClient.invalidateQueries({ queryKey: ['configs'] })
    },
    onError: (error: unknown) => {
      if (isAxiosError(error) && error.response?.status === 422) {
        const detail = error.response.data?.detail
        const issues = detail?.issues ?? []
        onError(
          [
            detail?.message ?? 'Validation failed.',
            ...issues.map((issue) => `${issue.path}: ${issue.message}`),
          ].join('\n'),
        )
      } else {
        const message = error instanceof Error ? error.message : 'Failed to apply config.'
        onError(message)
      }
    },
  })

  // Structured mode validation
  const validateStructured = (params: {
    target: EditorTarget | null
    providerName: string
    providerResourceType: string
    providerType: string
    providerSettingsJson: string
    resourceName: string
    resourceType: string
    resourceProviderType: string
    resourceConfigJson: string
  }): boolean => {
    const errors: Record<string, string> = {}

    if (params.target?.type === 'provider') {
      if (!(params.providerName || params.target.name)) errors['provider.name'] = 'Name is required'
      if (!params.providerResourceType) errors['provider.resource_type'] = 'Resource type is required'
      if (!params.providerType) errors['provider.provider_type'] = 'Provider type is required'
      try {
        JSON.parse(params.providerSettingsJson || '{}')
      } catch {
        errors['provider.settings'] = 'Settings must be valid JSON'
      }
    } else if (params.target?.type === 'resource') {
      if (!(params.resourceName || params.target.name)) errors['resource.name'] = 'Name is required'
      if (!params.resourceType) errors['resource.resource_type'] = 'Resource type is required'
      if (!params.resourceProviderType) errors['resource.provider_type'] = 'Provider type is required'
      try {
        JSON.parse(params.resourceConfigJson || '{}')
      } catch {
        errors['resource.config'] = 'Config must be valid JSON'
      }
    }

    setStructuredErrors(errors)
    return Object.keys(errors).length === 0
  }

  // Structured mode diff mutation
  const computeStructuredDiff = useMutation<ConfigDiffResponse, Error, StructuredMutationParams>({
    mutationFn: async (params) => {
      let rendered: string

      if (params.target.type === 'provider') {
        const settings: Record<string, unknown> = JSON.parse(params.providerSettingsJson || '{}')
        const resp = await renderProviderContent({
          name: params.providerName || params.target.name,
          resource_type: params.providerResourceType,
          provider_type: params.providerType,
          description: 'Preview',
          settings,
        })
        rendered = resp.content
      } else {
        const cfg: Record<string, unknown> = JSON.parse(params.resourceConfigJson || '{}')
        const resp = await renderResourceContent({
          name: params.resourceName || params.target.name,
          resource_type: params.resourceType,
          provider_type: params.resourceProviderType,
          description: 'Preview',
          config: cfg,
        })
        rendered = resp.content
      }

      return await diffConfig({
        project_id: params.projectId,
        relative_path: params.target.relativePath,
        proposed_content: rendered,
      })
    },
    onSuccess: (result) => {
      setStructuredDiff(result)
      onError('') // Clear error
    },
    onError: (error) => {
      const message = error instanceof Error ? error.message : 'Failed to compute diff.'
      onError(message)
      setStructuredDiff(null)
    },
  })

  // Structured mode apply mutation
  const applyStructuredMutation = useMutation<ConfigApplyResponse, Error, StructuredMutationParams>({
    mutationFn: async (params) => {
      if (
        !validateStructured({
          target: params.target,
          providerName: params.providerName,
          providerResourceType: params.providerResourceType,
          providerType: params.providerType,
          providerSettingsJson: params.providerSettingsJson,
          resourceName: params.resourceName,
          resourceType: params.resourceType,
          resourceProviderType: params.resourceProviderType,
          resourceConfigJson: params.resourceConfigJson,
        })
      ) {
        throw new Error('Please fix validation errors.')
      }

      if (params.target.type === 'provider') {
        const settings: Record<string, unknown> = JSON.parse(params.providerSettingsJson || '{}')
        return await applyProviderStructured({
          project_id: params.projectId,
          name: params.providerName || params.target.name,
          resource_type: params.providerResourceType,
          provider_type: params.providerType,
          description: 'Updated via UI',
          settings,
        })
      } else {
        const cfg: Record<string, unknown> = JSON.parse(params.resourceConfigJson || '{}')
        return await applyResourceStructured({
          project_id: params.projectId,
          name: params.resourceName || params.target.name,
          resource_type: params.resourceType,
          provider_type: params.resourceProviderType,
          description: 'Updated via UI',
          config: cfg,
        })
      }
    },
    onSuccess: (result) => {
      onSuccess('Config applied successfully.')
      setDiffResult(null)
      setBaseHash(result.sha256_after)
      queryClient.invalidateQueries({ queryKey: ['configs'] })
    },
    onError: (error: unknown) => {
      const message = error instanceof Error ? error.message : 'Failed to apply config.'
      onError(message)
    },
  })

  return {
    diffMutation,
    applyMutation,
    structuredErrors,
    setStructuredErrors,
    structuredDiff,
    setStructuredDiff,
    computeStructuredDiff,
    applyStructuredMutation,
    validateStructured,
  }
}

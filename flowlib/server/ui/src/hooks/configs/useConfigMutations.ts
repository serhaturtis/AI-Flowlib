import { useState } from 'react'
import { useMutation, useQueryClient, UseMutationResult } from '@tanstack/react-query'
import {
  diffConfig,
  applyConfig,
  applyProviderStructured,
  applyResourceStructured,
  applyMessageSourceStructured,
  renderProviderContent,
  renderResourceContent,
  renderMessageSourceContent,
  ConfigDiffResponse,
  ConfigApplyResponse,
} from '../../services/configs'
import type { EditorTarget } from '../../components/configs/ConfigEditor/ConfigEditor'

// =============================================================================
// Type Definitions
// =============================================================================

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

// Raw mode mutation params
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

// =============================================================================
// Structured Mutation Params - Discriminated Union
// =============================================================================

interface ProviderMutationParams {
  type: 'provider'
  projectId: string
  target: EditorTarget
  name: string
  resourceType: string
  providerType: string
  settingsJson: string
}

interface ResourceMutationParams {
  type: 'resource'
  projectId: string
  target: EditorTarget
  name: string
  resourceType: string
  providerType: string
  configJson: string
}

interface MessageSourceMutationParams {
  type: 'message_source'
  projectId: string
  target: EditorTarget
  name: string
  sourceType: string
  enabled: boolean
  settingsJson: string
}

/**
 * Discriminated union for structured mutation parameters.
 * Each config type has its own well-defined contract.
 */
export type StructuredMutationParams =
  | ProviderMutationParams
  | ResourceMutationParams
  | MessageSourceMutationParams

// =============================================================================
// Validation
// =============================================================================

interface ValidationResult {
  valid: boolean
  errors: Record<string, string>
}

function validateProviderParams(params: ProviderMutationParams): ValidationResult {
  const errors: Record<string, string> = {}

  if (!params.name && !params.target.name) {
    errors['provider.name'] = 'Name is required'
  }
  if (!params.resourceType) {
    errors['provider.resource_type'] = 'Resource type is required'
  }
  if (!params.providerType) {
    errors['provider.provider_type'] = 'Provider type is required'
  }
  try {
    JSON.parse(params.settingsJson || '{}')
  } catch {
    errors['provider.settings'] = 'Settings must be valid JSON'
  }

  return { valid: Object.keys(errors).length === 0, errors }
}

function validateResourceParams(params: ResourceMutationParams): ValidationResult {
  const errors: Record<string, string> = {}

  if (!params.name && !params.target.name) {
    errors['resource.name'] = 'Name is required'
  }
  if (!params.resourceType) {
    errors['resource.resource_type'] = 'Resource type is required'
  }
  if (!params.providerType) {
    errors['resource.provider_type'] = 'Provider type is required'
  }
  try {
    JSON.parse(params.configJson || '{}')
  } catch {
    errors['resource.config'] = 'Config must be valid JSON'
  }

  return { valid: Object.keys(errors).length === 0, errors }
}

function validateMessageSourceParams(params: MessageSourceMutationParams): ValidationResult {
  const errors: Record<string, string> = {}

  if (!params.name && !params.target.name) {
    errors['message_source.name'] = 'Name is required'
  }
  if (!params.sourceType) {
    errors['message_source.source_type'] = 'Source type is required'
  }
  try {
    JSON.parse(params.settingsJson || '{}')
  } catch {
    errors['message_source.settings'] = 'Settings must be valid JSON'
  }

  return { valid: Object.keys(errors).length === 0, errors }
}

/**
 * Validate structured mutation params based on type.
 * Uses discriminated union for type-safe validation.
 */
function validateStructuredParams(params: StructuredMutationParams): ValidationResult {
  switch (params.type) {
    case 'provider':
      return validateProviderParams(params)
    case 'resource':
      return validateResourceParams(params)
    case 'message_source':
      return validateMessageSourceParams(params)
  }
}

// =============================================================================
// Mutation Handlers
// =============================================================================

async function renderStructuredContent(params: StructuredMutationParams): Promise<string> {
  switch (params.type) {
    case 'provider': {
      const settings: Record<string, unknown> = JSON.parse(params.settingsJson || '{}')
      const resp = await renderProviderContent({
        name: params.name || params.target.name,
        resource_type: params.resourceType,
        provider_type: params.providerType,
        description: 'Preview',
        settings,
      })
      return resp.content
    }
    case 'resource': {
      const config: Record<string, unknown> = JSON.parse(params.configJson || '{}')
      const resp = await renderResourceContent({
        name: params.name || params.target.name,
        resource_type: params.resourceType,
        provider_type: params.providerType,
        description: 'Preview',
        config,
      })
      return resp.content
    }
    case 'message_source': {
      const settings: Record<string, unknown> = JSON.parse(params.settingsJson || '{}')
      const resp = await renderMessageSourceContent({
        name: params.name || params.target.name,
        source_type: params.sourceType,
        enabled: params.enabled,
        settings,
      })
      return resp.content
    }
  }
}

async function applyStructuredConfig(params: StructuredMutationParams): Promise<ConfigApplyResponse> {
  switch (params.type) {
    case 'provider': {
      const settings: Record<string, unknown> = JSON.parse(params.settingsJson || '{}')
      return await applyProviderStructured({
        project_id: params.projectId,
        name: params.name || params.target.name,
        resource_type: params.resourceType,
        provider_type: params.providerType,
        description: 'Updated via UI',
        settings,
      })
    }
    case 'resource': {
      const config: Record<string, unknown> = JSON.parse(params.configJson || '{}')
      return await applyResourceStructured({
        project_id: params.projectId,
        name: params.name || params.target.name,
        resource_type: params.resourceType,
        provider_type: params.providerType,
        description: 'Updated via UI',
        config,
      })
    }
    case 'message_source': {
      const settings: Record<string, unknown> = JSON.parse(params.settingsJson || '{}')
      return await applyMessageSourceStructured({
        project_id: params.projectId,
        name: params.name || params.target.name,
        source_type: params.sourceType,
        enabled: params.enabled,
        settings,
      })
    }
  }
}

// =============================================================================
// Hook Interface
// =============================================================================

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
}

/**
 * Hook for managing config editor mutations.
 *
 * Features:
 * - Raw mode diff/apply mutations
 * - Structured mode diff/apply mutations with discriminated union params
 * - Type-safe validation logic
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

  // Structured mode diff mutation
  const computeStructuredDiff = useMutation<ConfigDiffResponse, Error, StructuredMutationParams>({
    mutationFn: async (params) => {
      const rendered = await renderStructuredContent(params)

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
      const validation = validateStructuredParams(params)
      if (!validation.valid) {
        setStructuredErrors(validation.errors)
        throw new Error('Please fix validation errors.')
      }

      return await applyStructuredConfig(params)
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
  }
}

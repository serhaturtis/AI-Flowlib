import { useState, useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import {
  createProviderConfig,
  createResourceConfig,
  createMessageSource,
  ProviderConfigSummary,
  ResourceConfigSummary,
  MessageSourceSummary,
  ConfigType,
} from '../../services/configs'
import { POLLING_INTERVALS } from '../../constants/polling'

// Type guard for Axios-like error responses
interface AxiosErrorResponse {
  response?: {
    status: number
    data?: {
      detail?:
        | string
        | {
            message?: string
            issues?: Array<{ path?: string; message?: string }>
          }
    }
  }
  message?: string
}

function isAxiosError(error: unknown): error is AxiosErrorResponse {
  return (
    typeof error === 'object' &&
    error !== null &&
    'response' in error &&
    typeof (error as AxiosErrorResponse).response === 'object'
  )
}

export interface UseCreateMutationResult {
  isCreating: boolean
  error: string | null
  success: string | null
  setError: (error: string | null) => void
  setSuccess: (success: string | null) => void
  create: (params: CreateParams) => Promise<void>
}

interface CreateParams {
  type: ConfigType
  projectId: string
  name: string
  providerResourceType: string
  providerType: string
  providerDescription: string
  providerSettingsJson: string
  resourceType: string
  resourceProviderType: string
  resourceDescription: string
  resourceConfigJson: string
  messageSourceType: string
  messageSourceEnabled: boolean
  messageSourceSettingsJson: string
  onSuccess: (result: {
    type: ConfigType
    name: string
    config: ProviderConfigSummary | ResourceConfigSummary | MessageSourceSummary
  }) => void
  onClose: () => void
  resetForm: () => void
}

/**
 * Hook for managing config creation mutations.
 *
 * Features:
 * - Provider and resource creation
 * - JSON validation
 * - Comprehensive error handling with server validation details
 * - Query invalidation
 * - Success notifications with auto-close
 *
 * @returns Mutation state and create handler
 */
export function useCreateMutation(): UseCreateMutationResult {
  const queryClient = useQueryClient()
  const [isCreating, setIsCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const timeoutRef = useRef<number | null>(null)

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  const create = async (params: CreateParams) => {
    // Validation
    if (!params.projectId) {
      setError('Select a project before creating configs.')
      return
    }
    if (!params.name.trim()) {
      setError('Name is required.')
      return
    }

    setError(null)
    setSuccess(null)

    try {
      setIsCreating(true)

      if (params.type === 'provider') {
        // Parse and validate provider settings JSON
        let settings: Record<string, unknown>
        try {
          settings = JSON.parse(params.providerSettingsJson || '{}')
        } catch {
          setError('Provider settings must be valid JSON.')
          return
        }

        // Create provider config
        await createProviderConfig({
          project_id: params.projectId,
          name: params.name.trim(),
          resource_type: params.providerResourceType,
          provider_type: params.providerType.trim(),
          description: params.providerDescription.trim() || 'Auto-generated provider config.',
          settings,
        })

        // Invalidate queries to refresh lists
        await queryClient.invalidateQueries({ queryKey: ['configs', 'providers', params.projectId] })
        await queryClient.invalidateQueries({ queryKey: ['configs', 'aliases', params.projectId] })

        setSuccess('Configuration created successfully.')

        // Notify parent of success
        const createdName = params.name.trim()
        params.resetForm()

        // Brief delay to show success message before closing
        // Clear any existing timeout first
        if (timeoutRef.current !== null) {
          window.clearTimeout(timeoutRef.current)
        }
        timeoutRef.current = window.setTimeout(() => {
          params.onClose()
          setSuccess(null)
          params.onSuccess({
            type: 'provider',
            name: createdName,
            config: {
              name: createdName,
              resource_type: params.providerResourceType,
              provider_type: params.providerType.trim(),
              settings,
            },
          })
          timeoutRef.current = null
        }, POLLING_INTERVALS.SUCCESS_AUTO_CLOSE) as unknown as number
      } else if (params.type === 'resource') {
        // Parse and validate resource config JSON
        let cfg: Record<string, unknown>
        try {
          cfg = JSON.parse(params.resourceConfigJson || '{}')
        } catch {
          setError('Resource config must be valid JSON.')
          return
        }

        // Create resource config
        await createResourceConfig({
          project_id: params.projectId,
          name: params.name.trim(),
          resource_type: params.resourceType,
          provider_type: params.resourceProviderType.trim(),
          description: params.resourceDescription.trim() || 'Auto-generated model config.',
          config: cfg,
        })

        // Invalidate queries to refresh lists
        await queryClient.invalidateQueries({ queryKey: ['configs', 'resources', params.projectId] })
        await queryClient.invalidateQueries({ queryKey: ['configs', 'aliases', params.projectId] })

        setSuccess('Configuration created successfully.')

        // Notify parent of success
        const createdName = params.name.trim()
        params.resetForm()

        // Brief delay to show success message before closing
        // Clear any existing timeout first
        if (timeoutRef.current !== null) {
          window.clearTimeout(timeoutRef.current)
        }
        timeoutRef.current = window.setTimeout(() => {
          params.onClose()
          setSuccess(null)
          params.onSuccess({
            type: 'resource',
            name: createdName,
            config: {
              name: createdName,
              resource_type: params.resourceType,
              metadata: {},
            },
          })
          timeoutRef.current = null
        }, POLLING_INTERVALS.SUCCESS_AUTO_CLOSE) as unknown as number
      } else if (params.type === 'message_source') {
        // Parse and validate message source settings JSON
        let settings: Record<string, unknown>
        try {
          settings = JSON.parse(params.messageSourceSettingsJson || '{}')
        } catch {
          setError('Message source settings must be valid JSON.')
          return
        }

        // Create message source config
        await createMessageSource({
          project_id: params.projectId,
          name: params.name.trim(),
          source_type: params.messageSourceType,
          enabled: params.messageSourceEnabled,
          settings,
        })

        // Invalidate queries to refresh lists
        await queryClient.invalidateQueries({ queryKey: ['configs', 'message-sources', params.projectId] })

        setSuccess('Message source created successfully.')

        // Notify parent of success
        const createdName = params.name.trim()
        params.resetForm()

        // Brief delay to show success message before closing
        if (timeoutRef.current !== null) {
          window.clearTimeout(timeoutRef.current)
        }
        timeoutRef.current = window.setTimeout(() => {
          params.onClose()
          setSuccess(null)
          params.onSuccess({
            type: 'message_source',
            name: createdName,
            config: {
              name: createdName,
              source_type: params.messageSourceType,
              enabled: params.messageSourceEnabled,
              settings,
            },
          })
          timeoutRef.current = null
        }, POLLING_INTERVALS.SUCCESS_AUTO_CLOSE) as unknown as number
      }
    } catch (err) {
      // Surface server-side validation detail if available
      if (isAxiosError(err)) {
        const status = err.response?.status
        const detail = err.response?.data?.detail

        if (status === 422) {
          // Validation error - show detailed issues
          const detailObj = typeof detail === 'object' ? detail : null
          const issues = detailObj && Array.isArray(detailObj.issues) ? detailObj.issues : []
          const composed = [
            detailObj?.message ?? 'Validation failed.',
            ...issues.map((i) => [i.path, i.message].filter(Boolean).join(': ')),
          ]
            .filter(Boolean)
            .join('\n')
          setError(composed || 'Validation failed (422).')
        } else if (status === 400) {
          const message =
            typeof detail === 'string'
              ? detail
              : typeof detail === 'object'
                ? detail?.message ?? 'Bad request (400). Please verify required fields and try again.'
                : 'Bad request (400). Please verify required fields and try again.'
          setError(message)
        } else {
          const detailObj = typeof detail === 'object' ? detail : null
          const message = detailObj?.message ?? err.message ?? 'Failed to create configuration.'
          setError(message)
        }
      } else {
        const message = err instanceof Error ? err.message : 'Failed to create configuration.'
        setError(message)
      }
    } finally {
      setIsCreating(false)
    }
  }

  return {
    isCreating,
    error,
    success,
    setError,
    setSuccess,
    create,
  }
}

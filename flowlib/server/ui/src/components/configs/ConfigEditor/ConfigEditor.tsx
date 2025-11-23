import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { ConfigDiffResponse } from '../../../services/configs'
import { useProviderEditorState } from '../../../hooks/configs/useProviderEditorState'
import { useResourceEditorState } from '../../../hooks/configs/useResourceEditorState'
import { useConfigMutations } from '../../../hooks/configs/useConfigMutations'
import { EditorModeSelector } from './EditorModeSelector'
import { RawEditorForm } from './RawEditorForm'
import { ProviderEditorForm } from './ProviderEditorForm'
import { ResourceEditorForm } from './ResourceEditorForm'
import { AgentEditorForm } from './AgentEditorForm'
import { DiffPreview } from './DiffPreview'

export interface EditorTarget {
  type: 'provider' | 'resource' | 'agent'
  name: string
  relativePath: string
  resourceType?: string
  providerType?: string
  data?: unknown
}

export interface ConfigEditorProps {
  projectId: string
  target: EditorTarget | null
  content: string
  onContentChange: (value: string) => void
  baseHash: string
  setBaseHash: (value: string) => void
  diffResult: ConfigDiffResponse | null
  setDiffResult: (value: ConfigDiffResponse | null) => void
  setSuccess: (value: string | null) => void
  setError: (value: string | null) => void
  queryClient: ReturnType<typeof useQueryClient>
}

/**
 * ConfigEditor - Dual-mode editor for provider and resource configurations
 *
 * Features:
 * - Raw mode: Direct JSON/Python editing with diff preview
 * - Structured mode: Schema-driven form builder with validation
 * - Rename/Delete operations
 * - Real-time schema loading based on resource type
 */
export function ConfigEditor({
  projectId,
  target,
  content,
  onContentChange,
  baseHash,
  setBaseHash,
  diffResult,
  setDiffResult,
  setSuccess,
  setError,
}: ConfigEditorProps) {
  const [editorMode, setEditorMode] = useState<'raw' | 'structured'>('structured')

  // Custom hooks for state management
  const providerState = useProviderEditorState(target, projectId)
  const resourceState = useResourceEditorState(target, projectId)
  const mutations = useConfigMutations(
    (msg) => {
      setSuccess(msg)
      setError(null)
    },
    (msg) => {
      setError(msg)
      setSuccess(null)
    },
    setDiffResult,
    setBaseHash,
  )

  if (!projectId || !target) {
    return <p>Select a provider or resource to edit.</p>
  }

  // Handlers for raw mode
  const handleRawDiff = () => {
    setSuccess(null)
    mutations.diffMutation.mutate({ projectId, target, content })
  }

  const handleRawApply = () => {
    setSuccess(null)
    mutations.applyMutation.mutate({ projectId, target, content, baseHash })
  }

  // Handlers for structured mode
  const handleStructuredDiff = () => {
    mutations.setStructuredErrors({})
    mutations.setStructuredDiff(null)
    mutations.computeStructuredDiff.mutate({
      projectId,
      target,
      providerName: providerState.name,
      providerResourceType: providerState.resourceType,
      providerType: providerState.providerType,
      providerSettingsJson: providerState.settingsJson,
      resourceName: resourceState.name,
      resourceType: resourceState.resourceType,
      resourceProviderType: resourceState.providerType,
      resourceConfigJson: resourceState.configJson,
    })
  }

  const handleStructuredApply = () => {
    mutations.setStructuredErrors({})
    mutations.applyStructuredMutation.mutate({
      projectId,
      target,
      providerName: providerState.name,
      providerResourceType: providerState.resourceType,
      providerType: providerState.providerType,
      providerSettingsJson: providerState.settingsJson,
      resourceName: resourceState.name,
      resourceType: resourceState.resourceType,
      resourceProviderType: resourceState.providerType,
      resourceConfigJson: resourceState.configJson,
    })
  }

  return (
    <div style={{ marginTop: '1rem' }}>
      <h3>
        Editing {target.type}: <code>{target.name}</code>
      </h3>

      <EditorModeSelector mode={editorMode} onModeChange={setEditorMode} />

      {editorMode === 'raw' ? (
        <>
          <RawEditorForm
            content={content}
            onContentChange={onContentChange}
            diffMutation={mutations.diffMutation}
            applyMutation={mutations.applyMutation}
            onDiff={handleRawDiff}
            onApply={handleRawApply}
          />
          <DiffPreview diffResult={diffResult} />
        </>
      ) : target.type === 'provider' ? (
        <ProviderEditorForm
          projectId={projectId}
          target={target}
          providerState={providerState}
          mutations={mutations}
          onSuccess={(msg) => {
            setSuccess(msg)
            setError(null)
          }}
          onError={(msg) => {
            setError(msg)
            setSuccess(null)
          }}
          onDiff={handleStructuredDiff}
          onApply={handleStructuredApply}
        />
      ) : target.type === 'resource' ? (
        <ResourceEditorForm
          projectId={projectId}
          target={target}
          resourceState={resourceState}
          mutations={mutations}
          onSuccess={(msg) => {
            setSuccess(msg)
            setError(null)
          }}
          onError={(msg) => {
            setError(msg)
            setSuccess(null)
          }}
          onDiff={handleStructuredDiff}
          onApply={handleStructuredApply}
        />
      ) : (
        <AgentEditorForm
          projectId={projectId}
          target={target}
          mutations={mutations}
          onSuccess={(msg) => {
            setSuccess(msg)
            setError(null)
          }}
          onError={(msg) => {
            setError(msg)
            setSuccess(null)
          }}
          onDiff={handleStructuredDiff}
          onApply={handleStructuredApply}
        />
      )}
    </div>
  )
}

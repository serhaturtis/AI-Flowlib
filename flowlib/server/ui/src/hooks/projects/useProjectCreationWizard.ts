/**
 * Hook for managing project creation wizard state
 *
 * Manages multi-step form for creating Flowlib projects with
 * choice between empty and guided setup with provider configuration.
 */

import { useState, useCallback } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { createProject } from '../../services/projects'

export type WizardStep =
  | 'basic'
  | 'setup-type'
  | 'fast-config'
  | 'llm-provider'
  | 'embedding-provider'
  | 'vector-db'
  | 'model-config'
  | 'agents-tools'
  | 'review'

export type SetupType = 'empty' | 'fast' | 'guided'

export type ProjectConfig = {
  name: string
  description: string
  setupType: SetupType
  agentNames: string
  toolCategories: string
}

export type FastConfig = {
  llmProviderType: string
  modelPath: string
}

export type ProviderConfig = {
  name: string
  providerType: string
  settingsJson: string
}

export type ModelConfig = {
  name: string
  providerType: string
  configJson: string
}

export type AgentConfig = {
  name: string
  configJson: string
}

export type WizardState = {
  project: ProjectConfig
  fastConfig: FastConfig | null
  llmProvider: ProviderConfig | null
  embeddingProvider: ProviderConfig | null
  vectorDb: ProviderConfig | null
  graphDb: ProviderConfig | null
  modelConfig: ModelConfig | null
  agentConfig: AgentConfig | null
}

const defaultProjectConfig: ProjectConfig = {
  name: '',
  description: '',
  setupType: 'guided',
  agentNames: '',
  toolCategories: '',
}

const defaultState: WizardState = {
  project: defaultProjectConfig,
  fastConfig: null,
  llmProvider: null,
  embeddingProvider: null,
  vectorDb: null,
  graphDb: null,
  modelConfig: null,
  agentConfig: null,
}

export interface UseProjectCreationWizardResult {
  // Current state
  currentStep: WizardStep
  state: WizardState
  error: string | null
  isCreating: boolean
  createdProjectId: string | null

  // Step navigation
  nextStep: () => void
  prevStep: () => void
  goToStep: (step: WizardStep) => void
  selectSetupTypeAndAdvance: (setupType: SetupType) => void

  // Configuration
  updateProject: (updates: Partial<ProjectConfig>) => void
  updateFastConfig: (config: FastConfig | null) => void
  updateLlmProvider: (config: ProviderConfig | null) => void
  updateEmbeddingProvider: (config: ProviderConfig | null) => void
  updateVectorDb: (config: ProviderConfig | null) => void
  updateGraphDb: (config: ProviderConfig | null) => void
  updateModelConfig: (config: ModelConfig | null) => void
  updateAgentConfig: (config: AgentConfig | null) => void
  validateConfig: () => boolean

  // Creation
  createProject: (onComplete: () => void) => Promise<void>
  reset: () => void
}

/**
 * Get the step order based on setup type
 */
function getStepOrder(setupType: SetupType): WizardStep[] {
  if (setupType === 'empty') {
    return ['basic', 'setup-type', 'review']
  }
  if (setupType === 'fast') {
    return ['basic', 'setup-type', 'fast-config', 'review']
  }
  return [
    'basic',
    'setup-type',
    'llm-provider',
    'embedding-provider',
    'vector-db',
    'model-config',
    'agents-tools',
    'review',
  ]
}

/**
 * Hook for managing the project creation wizard flow
 */
export function useProjectCreationWizard(): UseProjectCreationWizardResult {
  const [currentStep, setCurrentStep] = useState<WizardStep>('basic')
  const [state, setState] = useState<WizardState>(defaultState)
  const [error, setError] = useState<string | null>(null)
  const [createdProjectId, setCreatedProjectId] = useState<string | null>(null)

  const queryClient = useQueryClient()
  const projectMutation = useMutation({
    mutationFn: createProject,
    onSuccess: (data) => {
      setCreatedProjectId(data.id)
      queryClient.invalidateQueries({ queryKey: ['projects'] })
    },
  })

  // ========== STEP NAVIGATION ==========

  const nextStep = useCallback(() => {
    const stepOrder = getStepOrder(state.project.setupType)
    const currentIndex = stepOrder.indexOf(currentStep)

    if (currentIndex < stepOrder.length - 1) {
      setCurrentStep(stepOrder[currentIndex + 1])
    }
  }, [currentStep, state.project.setupType])

  const prevStep = useCallback(() => {
    const stepOrder = getStepOrder(state.project.setupType)
    const currentIndex = stepOrder.indexOf(currentStep)

    if (currentIndex > 0) {
      setCurrentStep(stepOrder[currentIndex - 1])
    }
  }, [currentStep, state.project.setupType])

  const goToStep = useCallback((step: WizardStep) => {
    setCurrentStep(step)
  }, [])

  /**
   * Select setup type and advance to next step.
   * This handles the state update properly by using the new setupType
   * to determine the next step, avoiding race conditions.
   */
  const selectSetupTypeAndAdvance = useCallback((setupType: SetupType) => {
    // Update the setup type
    setState((prev) => ({
      ...prev,
      project: { ...prev.project, setupType },
    }))
    setError(null)

    // Determine next step based on the NEW setupType
    const stepOrder = getStepOrder(setupType)
    const currentIndex = stepOrder.indexOf(currentStep)

    if (currentIndex < stepOrder.length - 1) {
      setCurrentStep(stepOrder[currentIndex + 1])
    }
  }, [currentStep])

  // ========== CONFIGURATION ==========

  const updateProject = useCallback((updates: Partial<ProjectConfig>) => {
    setState((prev) => ({
      ...prev,
      project: { ...prev.project, ...updates },
    }))
    setError(null)
  }, [])

  const updateFastConfig = useCallback((config: FastConfig | null) => {
    setState((prev) => ({ ...prev, fastConfig: config }))
    setError(null)
  }, [])

  const updateLlmProvider = useCallback((config: ProviderConfig | null) => {
    setState((prev) => ({ ...prev, llmProvider: config }))
    setError(null)
  }, [])

  const updateEmbeddingProvider = useCallback((config: ProviderConfig | null) => {
    setState((prev) => ({ ...prev, embeddingProvider: config }))
    setError(null)
  }, [])

  const updateVectorDb = useCallback((config: ProviderConfig | null) => {
    setState((prev) => ({ ...prev, vectorDb: config }))
    setError(null)
  }, [])

  const updateGraphDb = useCallback((config: ProviderConfig | null) => {
    setState((prev) => ({ ...prev, graphDb: config }))
    setError(null)
  }, [])

  const updateModelConfig = useCallback((config: ModelConfig | null) => {
    setState((prev) => ({ ...prev, modelConfig: config }))
    setError(null)
  }, [])

  const updateAgentConfig = useCallback((config: AgentConfig | null) => {
    setState((prev) => ({ ...prev, agentConfig: config }))
    setError(null)
  }, [])

  const validateConfig = useCallback((): boolean => {
    const { project, fastConfig, llmProvider, embeddingProvider } = state

    if (!project.name.trim()) {
      setError('Project name is required')
      return false
    }

    if (project.name.length < 3) {
      setError('Project name must be at least 3 characters')
      return false
    }

    if (project.name.length > 128) {
      setError('Project name must be less than 128 characters')
      return false
    }

    // For fast setup, validate model path
    if (project.setupType === 'fast') {
      if (!fastConfig || !fastConfig.modelPath.trim()) {
        setError('Model path is required for fast setup')
        return false
      }
    }

    // For guided setup, validate provider configs
    if (project.setupType === 'guided') {
      if (!llmProvider) {
        setError('LLM provider configuration is required for guided setup')
        return false
      }

      if (!embeddingProvider) {
        setError('Embedding provider configuration is required for guided setup')
        return false
      }
    }

    setError(null)
    return true
  }, [state])

  // ========== RESET ==========

  const reset = useCallback(() => {
    setCurrentStep('basic')
    setState(defaultState)
    setError(null)
    setCreatedProjectId(null)
  }, [])

  // ========== CREATION ==========

  const createProjectHandler = useCallback(
    async (onComplete: () => void) => {
      if (!validateConfig()) {
        return
      }

      try {
        setError(null)

        const agentNames = state.project.agentNames
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean)

        const toolCategories = state.project.toolCategories
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean)

        // Build the atomic project creation request
        const createRequest: Parameters<typeof createProject>[0] = {
          name: state.project.name,
          description: state.project.description || 'Project created via Flowlib wizard.',
          setup_type: state.project.setupType,
          agent_names: agentNames,
          tool_categories: toolCategories,
        }

        // Add setup-specific configuration
        if (state.project.setupType === 'fast' && state.fastConfig) {
          createRequest.fast_config = {
            llm_model_path: state.fastConfig.modelPath,
            embedding_model_path: state.fastConfig.modelPath, // TODO: Add separate embedding path field
            vector_db_path: null,
          }
        } else if (state.project.setupType === 'guided') {
          const providers: Array<{
            name: string
            resource_type: string
            provider_type: string
            description?: string
            settings?: Record<string, unknown>
          }> = []
          const resources: Array<{
            name: string
            resource_type: string
            provider_type: string
            description?: string
            config?: Record<string, unknown>
          }> = []
          const aliases: Record<string, string> = {}

          // Build provider configs
          if (state.llmProvider) {
            providers.push({
              name: state.llmProvider.name,
              resource_type: 'llm_config',
              provider_type: state.llmProvider.providerType,
              description: 'LLM provider',
              settings: JSON.parse(state.llmProvider.settingsJson || '{}'),
            })
            aliases['default-llm'] = state.llmProvider.name
          }

          if (state.embeddingProvider) {
            providers.push({
              name: state.embeddingProvider.name,
              resource_type: 'embedding_config',
              provider_type: state.embeddingProvider.providerType,
              description: 'Embedding provider',
              settings: JSON.parse(state.embeddingProvider.settingsJson || '{}'),
            })
            aliases['default-embedding'] = state.embeddingProvider.name
          }

          if (state.vectorDb) {
            providers.push({
              name: state.vectorDb.name,
              resource_type: 'vector_db_config',
              provider_type: state.vectorDb.providerType,
              description: 'Vector database',
              settings: JSON.parse(state.vectorDb.settingsJson || '{}'),
            })
            aliases['default-vector-db'] = state.vectorDb.name
          }

          if (state.graphDb) {
            providers.push({
              name: state.graphDb.name,
              resource_type: 'graph_db_config',
              provider_type: state.graphDb.providerType,
              description: 'Graph database',
              settings: JSON.parse(state.graphDb.settingsJson || '{}'),
            })
            aliases['default-graph-db'] = state.graphDb.name
          }

          // Build resource configs
          if (state.modelConfig) {
            resources.push({
              name: state.modelConfig.name,
              resource_type: 'model_config',
              provider_type: state.modelConfig.providerType,
              description: 'Model configuration',
              config: JSON.parse(state.modelConfig.configJson || '{}'),
            })
            aliases['default-model'] = state.modelConfig.name
          }

          createRequest.guided_config = {
            providers,
            resources,
            aliases,
          }
        }

        // Single atomic project creation call
        const project = await projectMutation.mutateAsync(createRequest)

        // Invalidate all config queries for the new project
        queryClient.invalidateQueries({ queryKey: ['configs', project.id] })

        // Success!
        onComplete()
        reset()
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to create project'
        setError(message)
        throw err
      }
    },
    [state, validateConfig, projectMutation, queryClient, reset],
  )

  return {
    // Current state
    currentStep,
    state,
    error,
    isCreating: projectMutation.isPending,
    createdProjectId,

    // Step navigation
    nextStep,
    prevStep,
    goToStep,
    selectSetupTypeAndAdvance,

    // Configuration
    updateProject,
    updateFastConfig,
    updateLlmProvider,
    updateEmbeddingProvider,
    updateVectorDb,
    updateGraphDb,
    updateModelConfig,
    updateAgentConfig,
    validateConfig,

    // Creation
    createProject: createProjectHandler,
    reset,
  }
}

/**
 * Hook for managing plugin generation wizard state
 *
 * Manages multi-step form for creating knowledge plugins with file upload,
 * configuration, and generation progress tracking.
 */

import { useState, useCallback } from 'react'
import { DomainStrategy } from '../../services/knowledge'
import { useGeneratePlugin, useUploadDocuments } from './useKnowledgePlugins'

/**
 * Validation constraints for plugin generation
 */
const PLUGIN_VALIDATION_LIMITS = {
  CHUNK_SIZE_MIN: 100,
  CHUNK_SIZE_MAX: 5000,
  CHUNK_OVERLAP_MIN: 0,
  CHUNK_OVERLAP_MAX: 1000,
} as const

export type WizardStep = 'upload' | 'configure' | 'generate' | 'complete'

export type PluginConfig = {
  plugin_name: string
  domains: string[]
  description: string
  author: string
  use_vector_db: boolean
  use_graph_db: boolean
  chunk_size: number
  chunk_overlap: number
  domain_strategy: DomainStrategy
}

const defaultConfig: PluginConfig = {
  plugin_name: '',
  domains: [],
  description: '',
  author: 'Knowledge Plugin Generator',
  use_vector_db: true,
  use_graph_db: true,
  chunk_size: 1000,
  chunk_overlap: 200,
  domain_strategy: DomainStrategy.GENERIC,
}

export interface UsePluginGenerationWizardResult {
  // Current state
  currentStep: WizardStep
  config: PluginConfig
  uploadedFiles: File[]
  uploadDirectory: string | null
  error: string | null
  isUploading: boolean
  isGenerating: boolean
  generationProgress: number

  // Step navigation
  nextStep: () => void
  prevStep: () => void
  goToStep: (step: WizardStep) => void

  // File management
  handleFilesSelected: (files: File[]) => void
  removeFile: (index: number) => void
  uploadFiles: (projectId: string) => Promise<void>

  // Configuration
  updateConfig: (updates: Partial<PluginConfig>) => void
  validateConfig: () => boolean

  // Generation
  generatePlugin: (projectId: string) => Promise<void>
  reset: () => void
}

/**
 * Hook for managing the plugin generation wizard flow
 */
export function usePluginGenerationWizard(): UsePluginGenerationWizardResult {
  const [currentStep, setCurrentStep] = useState<WizardStep>('upload')
  const [config, setConfig] = useState<PluginConfig>(defaultConfig)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [uploadDirectory, setUploadDirectory] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [generationProgress, setGenerationProgress] = useState(0)

  const uploadMutation = useUploadDocuments()
  const generateMutation = useGeneratePlugin()

  // ========== STEP NAVIGATION ==========

  const nextStep = useCallback(() => {
    const stepOrder: WizardStep[] = ['upload', 'configure', 'generate', 'complete']
    const currentIndex = stepOrder.indexOf(currentStep)
    if (currentIndex < stepOrder.length - 1) {
      setCurrentStep(stepOrder[currentIndex + 1])
    }
  }, [currentStep])

  const prevStep = useCallback(() => {
    const stepOrder: WizardStep[] = ['upload', 'configure', 'generate', 'complete']
    const currentIndex = stepOrder.indexOf(currentStep)
    if (currentIndex > 0) {
      setCurrentStep(stepOrder[currentIndex - 1])
    }
  }, [currentStep])

  const goToStep = useCallback((step: WizardStep) => {
    setCurrentStep(step)
  }, [])

  // ========== FILE MANAGEMENT ==========

  const handleFilesSelected = useCallback((files: File[]) => {
    setUploadedFiles(files)
    setError(null)
  }, [])

  const removeFile = useCallback((index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index))
  }, [])

  const uploadFiles = useCallback(
    async (projectId: string) => {
      if (uploadedFiles.length === 0) {
        setError('Please select at least one file to upload')
        return
      }

      try {
        setError(null)
        const result = await uploadMutation.mutateAsync({ projectId, files: uploadedFiles })
        setUploadDirectory(result.upload_directory)
        nextStep()
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to upload files'
        setError(message)
        throw err
      }
    },
    [uploadedFiles, uploadMutation, nextStep],
  )

  // ========== CONFIGURATION ==========

  const updateConfig = useCallback((updates: Partial<PluginConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }))
    setError(null)
  }, [])

  const validateConfig = useCallback((): boolean => {
    if (!config.plugin_name.trim()) {
      setError('Plugin name is required')
      return false
    }

    // Validate plugin name format (lowercase, alphanumeric, hyphens, underscores)
    if (!/^[a-z0-9_-]+$/.test(config.plugin_name)) {
      setError('Plugin name must contain only lowercase letters, numbers, hyphens, and underscores')
      return false
    }

    if (config.domains.length === 0) {
      setError('At least one domain is required')
      return false
    }

    if (config.chunk_size < PLUGIN_VALIDATION_LIMITS.CHUNK_SIZE_MIN || config.chunk_size > PLUGIN_VALIDATION_LIMITS.CHUNK_SIZE_MAX) {
      setError(`Chunk size must be between ${PLUGIN_VALIDATION_LIMITS.CHUNK_SIZE_MIN} and ${PLUGIN_VALIDATION_LIMITS.CHUNK_SIZE_MAX}`)
      return false
    }

    if (config.chunk_overlap < PLUGIN_VALIDATION_LIMITS.CHUNK_OVERLAP_MIN || config.chunk_overlap > PLUGIN_VALIDATION_LIMITS.CHUNK_OVERLAP_MAX) {
      setError(`Chunk overlap must be between ${PLUGIN_VALIDATION_LIMITS.CHUNK_OVERLAP_MIN} and ${PLUGIN_VALIDATION_LIMITS.CHUNK_OVERLAP_MAX}`)
      return false
    }

    setError(null)
    return true
  }, [config])

  // ========== GENERATION ==========

  const generatePlugin = useCallback(
    async (projectId: string) => {
      if (!uploadDirectory) {
        setError('No upload directory available. Please upload files first.')
        return
      }

      if (!validateConfig()) {
        return
      }

      try {
        setError(null)
        setGenerationProgress(10)

        const result = await generateMutation.mutateAsync({
          project_id: projectId,
          plugin_name: config.plugin_name,
          input_directory: uploadDirectory,
          domains: config.domains,
          description: config.description || undefined,
          author: config.author,
          use_vector_db: config.use_vector_db,
          use_graph_db: config.use_graph_db,
          chunk_size: config.chunk_size,
          chunk_overlap: config.chunk_overlap,
          domain_strategy: config.domain_strategy,
        })

        if (!result.success) {
          throw new Error(result.error_message || 'Plugin generation failed')
        }

        setGenerationProgress(100)
        nextStep()
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to generate plugin'
        setError(message)
        setGenerationProgress(0)
        throw err
      }
    },
    [uploadDirectory, config, validateConfig, generateMutation, nextStep],
  )

  const reset = useCallback(() => {
    setCurrentStep('upload')
    setConfig(defaultConfig)
    setUploadedFiles([])
    setUploadDirectory(null)
    setError(null)
    setGenerationProgress(0)
  }, [])

  return {
    // Current state
    currentStep,
    config,
    uploadedFiles,
    uploadDirectory,
    error,
    isUploading: uploadMutation.isPending,
    isGenerating: generateMutation.isPending,
    generationProgress,

    // Step navigation
    nextStep,
    prevStep,
    goToStep,

    // File management
    handleFilesSelected,
    removeFile,
    uploadFiles,

    // Configuration
    updateConfig,
    validateConfig,

    // Generation
    generatePlugin,
    reset,
  }
}

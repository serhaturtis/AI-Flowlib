/**
 * Project Creation Wizard Component
 *
 * Multi-step wizard for creating Flowlib projects:
 * - Empty Setup: Basic → Setup Type → Review
 * - Guided Setup: Basic → Setup Type → LLM → Embedding → Vector/Graph DB → Model → Agents/Tools → Review
 */

import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Label } from '../ui/Label'
import { Textarea } from '../ui/Textarea'
import { Alert, AlertDescription } from '../ui/Alert'
import { Separator } from '../ui/Separator'
import {
  FolderOpen,
  Settings,
  Loader2,
  CheckCircle2,
  ArrowRight,
  ArrowLeft,
  FileCode,
  Wand2,
  Cpu,
  Database,
  Brain,
  AlertCircle,
} from 'lucide-react'
import { useProjectCreationWizard, type WizardStep } from '../../hooks/projects/useProjectCreationWizard'
import { listProviderTypes, fetchProviderSchema, fetchResourceSchema, fetchAgentSchema } from '../../services/configs'
import { fetchProjects } from '../../services/projects'
import { SchemaFieldInput } from '../configs/SchemaFields/SchemaFieldInput'
import { Spinner } from '../ui/Spinner'

type Props = {
  onComplete: () => void
  onCancel: () => void
}

// Step metadata for progress indicator
const stepMetadata: Record<WizardStep, { label: string; shortLabel?: string }> = {
  basic: { label: 'Basic Info' },
  'setup-type': { label: 'Setup Type', shortLabel: 'Setup' },
  'fast-config': { label: 'Quick Config', shortLabel: 'Config' },
  'llm-provider': { label: 'LLM Provider', shortLabel: 'LLM' },
  'embedding-provider': { label: 'Embedding', shortLabel: 'Embed' },
  'vector-db': { label: 'Databases', shortLabel: 'DB' },
  'model-config': { label: 'Model Config', shortLabel: 'Model' },
  'agents-tools': { label: 'Agents & Tools', shortLabel: 'Agents' },
  review: { label: 'Review' },
}

export function ProjectCreationWizard({ onComplete, onCancel }: Props) {
  const wizard = useProjectCreationWizard()

  // Destructure wizard methods for stable references in useEffect dependencies
  const {
    updateFastConfig,
    updateLlmProvider,
    updateEmbeddingProvider,
    updateModelConfig,
    updateAgentConfig,
  } = wizard

  // Fetch existing projects to use an existing project ID for schema queries
  // (schemas are the same across projects, we just need any valid project ID)
  const { data: projectsData } = useQuery({
    queryKey: ['projects'],
    queryFn: fetchProjects,
  })

  // Use first existing project's ID for schema fetching, or fallback to 'temp'
  const schemaProjectId = projectsData?.projects?.[0]?.id || 'temp'

  // Fetch available provider types from registry
  const { data: llmProviderTypes = [] } = useQuery({
    queryKey: ['providerTypes', 'llm_config'],
    queryFn: () => listProviderTypes('llm_config'),
  })

  const { data: embeddingProviderTypes = [] } = useQuery({
    queryKey: ['providerTypes', 'embedding_config'],
    queryFn: () => listProviderTypes('embedding_config'),
  })

  const { data: vectorDbProviderTypes = [] } = useQuery({
    queryKey: ['providerTypes', 'vector_db_config'],
    queryFn: () => listProviderTypes('vector_db_config'),
  })

  const { data: graphDbProviderTypes = [] } = useQuery({
    queryKey: ['providerTypes', 'graph_db_config'],
    queryFn: () => listProviderTypes('graph_db_config'),
  })

  // Fetch schemas for provider configuration
  const { data: llmSchema, isLoading: llmSchemaLoading, error: llmSchemaError } = useQuery({
    queryKey: ['providerSchema', 'llm_config', wizard.state.llmProvider?.providerType, schemaProjectId],
    queryFn: () => fetchProviderSchema('llm_config', schemaProjectId, wizard.state.llmProvider?.providerType),
    enabled: wizard.currentStep === 'llm-provider' && !!wizard.state.llmProvider?.providerType && !!schemaProjectId,
  })

  const { data: embeddingSchema, isLoading: embeddingSchemaLoading, error: embeddingSchemaError } = useQuery({
    queryKey: ['providerSchema', 'embedding_config', wizard.state.embeddingProvider?.providerType, schemaProjectId],
    queryFn: () => fetchProviderSchema('embedding_config', schemaProjectId, wizard.state.embeddingProvider?.providerType),
    enabled: wizard.currentStep === 'embedding-provider' && !!wizard.state.embeddingProvider?.providerType && !!schemaProjectId,
  })

  const { data: vectorDbSchema, isLoading: vectorDbSchemaLoading, error: vectorDbSchemaError } = useQuery({
    queryKey: ['providerSchema', 'vector_db_config', wizard.state.vectorDb?.providerType, schemaProjectId],
    queryFn: () => fetchProviderSchema('vector_db_config', schemaProjectId, wizard.state.vectorDb?.providerType),
    enabled: wizard.currentStep === 'vector-db' && !!wizard.state.vectorDb?.providerType && !!schemaProjectId,
  })

  const { data: graphDbSchema, isLoading: graphDbSchemaLoading, error: graphDbSchemaError } = useQuery({
    queryKey: ['providerSchema', 'graph_db_config', wizard.state.graphDb?.providerType, schemaProjectId],
    queryFn: () => fetchProviderSchema('graph_db_config', schemaProjectId, wizard.state.graphDb?.providerType),
    enabled: wizard.currentStep === 'vector-db' && !!wizard.state.graphDb?.providerType && !!schemaProjectId,
  })

  const { data: modelConfigSchema, isLoading: modelConfigSchemaLoading, error: modelConfigSchemaError } = useQuery({
    queryKey: ['resourceSchema', 'model_config', wizard.state.modelConfig?.providerType, schemaProjectId],
    queryFn: () => fetchResourceSchema('model_config', schemaProjectId, wizard.state.modelConfig?.providerType),
    enabled: wizard.currentStep === 'model-config' && !!wizard.state.modelConfig?.providerType && !!schemaProjectId,
  })

  const { data: agentSchema, isLoading: agentSchemaLoading, error: agentSchemaError } = useQuery({
    queryKey: ['agentSchema', schemaProjectId],
    queryFn: () => fetchAgentSchema(schemaProjectId),
    enabled: wizard.currentStep === 'agents-tools' && !!schemaProjectId,
  })

  // Auto-initialize provider states when entering steps
  useEffect(() => {
    if (wizard.currentStep === 'fast-config' && !wizard.state.fastConfig && llmProviderTypes.length > 0) {
      // Default to llamacpp for fast setup (most common for local models)
      const defaultProvider = llmProviderTypes.includes('llamacpp') ? 'llamacpp' : llmProviderTypes[0]
      updateFastConfig({
        llmProviderType: defaultProvider,
        modelPath: '',
      })
    }
  }, [wizard.currentStep, wizard.state.fastConfig, llmProviderTypes, updateFastConfig])

  useEffect(() => {
    if (wizard.currentStep === 'llm-provider' && !wizard.state.llmProvider && llmProviderTypes.length > 0) {
      updateLlmProvider({
        name: 'default-llm',
        providerType: llmProviderTypes[0],
        settingsJson: '{}',
      })
    }
  }, [wizard.currentStep, wizard.state.llmProvider, llmProviderTypes, updateLlmProvider])

  useEffect(() => {
    if (wizard.currentStep === 'embedding-provider' && !wizard.state.embeddingProvider && embeddingProviderTypes.length > 0) {
      updateEmbeddingProvider({
        name: 'default-embedding',
        providerType: embeddingProviderTypes[0],
        settingsJson: '{}',
      })
    }
  }, [wizard.currentStep, wizard.state.embeddingProvider, embeddingProviderTypes, updateEmbeddingProvider])

  useEffect(() => {
    if (wizard.currentStep === 'model-config' && !wizard.state.modelConfig && wizard.state.llmProvider?.providerType) {
      updateModelConfig({
        name: 'default-model',
        providerType: wizard.state.llmProvider.providerType,
        configJson: '{}',
      })
    }
  }, [wizard.currentStep, wizard.state.modelConfig, wizard.state.llmProvider, updateModelConfig])

  useEffect(() => {
    if (wizard.currentStep === 'agents-tools' && !wizard.state.agentConfig) {
      updateAgentConfig({
        name: 'default-agent',
        configJson: JSON.stringify({
          persona: '',
          allowed_tool_categories: [],
          model_name: 'default-model',
          llm_name: 'default-llm',
          temperature: 0.7,
          max_iterations: 10,
          enable_learning: true,
        }),
      })
    }
  }, [wizard.currentStep, wizard.state.agentConfig, updateAgentConfig])

  // Get step order based on setup type
  const stepOrder: WizardStep[] =
    wizard.state.project.setupType === 'empty'
      ? ['basic', 'setup-type', 'review']
      : ['basic', 'setup-type', 'llm-provider', 'embedding-provider', 'vector-db', 'model-config', 'agents-tools', 'review']

  const currentIndex = stepOrder.indexOf(wizard.currentStep)

  return (
    <div className="space-y-6">
      {/* Progress Steps */}
      <div className="flex items-center justify-between mb-8">
        {stepOrder.map((step, index) => {
          const isActive = index === currentIndex
          const isCompleted = index < currentIndex
          const metadata = stepMetadata[step]
          const displayLabel = stepOrder.length > 5 && metadata.shortLabel ? metadata.shortLabel : metadata.label

          return (
            <div key={step} className="flex items-center flex-1">
              <div className="flex flex-col items-center flex-1">
                <div
                  className={`
                    w-10 h-10 rounded-full flex items-center justify-center border-2
                    ${isActive ? 'border-primary bg-primary text-primary-foreground' : ''}
                    ${isCompleted ? 'border-primary bg-primary text-primary-foreground' : ''}
                    ${!isActive && !isCompleted ? 'border-muted bg-muted text-muted-foreground' : ''}
                  `}
                >
                  {isCompleted ? <CheckCircle2 className="h-5 w-5" /> : index + 1}
                </div>
                <span className="text-xs mt-2 font-medium text-center">{displayLabel}</span>
              </div>
              {index < stepOrder.length - 1 && (
                <div
                  className={`h-0.5 flex-1 mx-2 ${isCompleted ? 'bg-primary' : 'bg-muted'}`}
                />
              )}
            </div>
          )
        })}
      </div>

      {/* Error Display */}
      {wizard.error && (
        <Alert variant="destructive">
          <AlertDescription>{wizard.error}</AlertDescription>
        </Alert>
      )}

      {/* Step 1: Basic Information */}
      {wizard.currentStep === 'basic' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2">Basic Information</h3>
            <p className="text-sm text-muted-foreground">
              Enter the basic details for your new Flowlib project
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <Label htmlFor="project-name">Project Name *</Label>
              <Input
                id="project-name"
                value={wizard.state.project.name}
                onChange={(e) => wizard.updateProject({ name: e.target.value })}
                placeholder="my-awesome-project"
                className="mt-1"
              />
              <p className="text-xs text-muted-foreground mt-1">
                This will be used as the project directory name
              </p>
            </div>

            <div>
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={wizard.state.project.description}
                onChange={(e) => wizard.updateProject({ description: e.target.value })}
                placeholder="Describe what this project is for..."
                className="mt-1"
                rows={3}
              />
            </div>
          </div>

          <div className="flex justify-between pt-4">
            <Button type="button" variant="outline" onClick={onCancel}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={wizard.nextStep}
              disabled={!wizard.state.project.name.trim()}
            >
              Next
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </div>
      )}

      {/* Step 2: Setup Type */}
      {wizard.currentStep === 'setup-type' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2">Choose Setup Type</h3>
            <p className="text-sm text-muted-foreground">
              Select how you want to set up your project
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Empty Project Option */}
            <button
              type="button"
              onClick={() => wizard.selectSetupTypeAndAdvance('empty')}
              className={`
                p-6 border-2 rounded-lg text-left transition-all hover:border-primary
                ${
                  wizard.state.project.setupType === 'empty'
                    ? 'border-primary bg-primary/5'
                    : 'border-muted'
                }
              `}
            >
              <FileCode className="h-8 w-8 mb-3 text-muted-foreground" />
              <h4 className="font-semibold mb-2">Empty Project</h4>
              <p className="text-sm text-muted-foreground">
                Create a bare project structure. You'll manually configure providers, models,
                and aliases.
              </p>
              <ul className="mt-3 text-sm text-muted-foreground space-y-1">
                <li>• Basic directory structure</li>
                <li>• Example configuration templates</li>
                <li>• Full manual control</li>
              </ul>
            </button>

            {/* Fast Setup Option */}
            <button
              type="button"
              onClick={() => wizard.selectSetupTypeAndAdvance('fast')}
              className={`
                p-6 border-2 rounded-lg text-left transition-all hover:border-primary
                ${
                  wizard.state.project.setupType === 'fast'
                    ? 'border-primary bg-primary/5'
                    : 'border-muted'
                }
              `}
            >
              <Settings className="h-8 w-8 mb-3 text-accent" />
              <h4 className="font-semibold mb-2">Fast Setup</h4>
              <p className="text-sm text-muted-foreground">
                Get started quickly with sensible defaults. Just provide your LLM model path and go.
              </p>
              <ul className="mt-3 text-sm text-muted-foreground space-y-1">
                <li>• Minimal configuration required</li>
                <li>• Sensible defaults for everything</li>
                <li>• Ready to run immediately</li>
                <li>• Perfect for quick testing</li>
              </ul>
            </button>

            {/* Guided Setup Option */}
            <button
              type="button"
              onClick={() => wizard.selectSetupTypeAndAdvance('guided')}
              className={`
                p-6 border-2 rounded-lg text-left transition-all hover:border-primary
                ${
                  wizard.state.project.setupType === 'guided'
                    ? 'border-primary bg-primary/5'
                    : 'border-muted'
                }
              `}
            >
              <Wand2 className="h-8 w-8 mb-3 text-primary" />
              <h4 className="font-semibold mb-2">Guided Setup</h4>
              <p className="text-sm text-muted-foreground">
                Let us help you set up providers and models. We'll create working configurations
                for you.
              </p>
              <ul className="mt-3 text-sm text-muted-foreground space-y-1">
                <li>• Choose your LLM provider</li>
                <li>• Configure embedding models</li>
                <li>• Set up vector and graph databases</li>
                <li>• Create default aliases</li>
                <li>• Ready to use immediately</li>
              </ul>
            </button>
          </div>

          <div className="flex justify-between pt-4">
            <Button type="button" variant="outline" onClick={wizard.prevStep}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
          </div>
        </div>
      )}

      {/* Step 3: Fast Config */}
      {wizard.currentStep === 'fast-config' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Quick Configuration
            </h3>
            <p className="text-sm text-muted-foreground">
              Provide essential information to create a runnable project with defaults
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <Label htmlFor="fast-llm-provider">LLM Provider Type *</Label>
              <select
                id="fast-llm-provider"
                value={wizard.state.fastConfig?.llmProviderType || (llmProviderTypes.includes('llamacpp') ? 'llamacpp' : llmProviderTypes[0] || '')}
                onChange={(e) =>
                  wizard.updateFastConfig({
                    llmProviderType: e.target.value,
                    modelPath: wizard.state.fastConfig?.modelPath || '',
                  })
                }
                className="w-full mt-1 px-3 py-2 border border-input rounded-md"
                disabled={llmProviderTypes.length === 0}
              >
                {llmProviderTypes.length === 0 ? (
                  <option value="">Loading providers...</option>
                ) : (
                  llmProviderTypes.map((providerType) => (
                    <option key={providerType} value={providerType}>
                      {providerType}{providerType === 'llamacpp' ? ' (recommended for local models)' : ''}
                    </option>
                  ))
                )}
              </select>
              <p className="text-xs text-muted-foreground mt-1">
                Defaults to llamacpp for local GGUF models
              </p>
            </div>

            <div>
              <Label htmlFor="fast-model-path">Model Path *</Label>
              <Input
                id="fast-model-path"
                value={wizard.state.fastConfig?.modelPath || ''}
                onChange={(e) =>
                  wizard.updateFastConfig({
                    llmProviderType: wizard.state.fastConfig?.llmProviderType || (llmProviderTypes.includes('llamacpp') ? 'llamacpp' : llmProviderTypes[0] || ''),
                    modelPath: e.target.value,
                  })
                }
                placeholder="/path/to/your/model.gguf"
                className="mt-1"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Absolute path to your GGUF model file (e.g., /home/user/models/llama-2-7b-chat.Q4_K_M.gguf)
              </p>
            </div>

            <div className="p-4 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg">
              <h4 className="font-medium text-sm mb-2 text-blue-900 dark:text-blue-100">What will be created:</h4>
              <ul className="text-xs text-blue-800 dark:text-blue-200 space-y-1">
                <li>• LLM provider with your model path</li>
                <li>• Embedding provider (auto-configured)</li>
                <li>• Default model configuration</li>
                <li>• Default agent configuration</li>
                <li>• All necessary aliases</li>
              </ul>
            </div>
          </div>

          <div className="flex justify-between pt-4">
            <Button type="button" variant="outline" onClick={wizard.prevStep}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button
              type="button"
              onClick={wizard.nextStep}
              disabled={!wizard.state.fastConfig?.modelPath.trim()}
            >
              Next
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </div>
      )}

      {/* Step 4: LLM Provider */}
      {wizard.currentStep === 'llm-provider' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
              <Brain className="h-5 w-5" />
              LLM Provider Configuration
            </h3>
            <p className="text-sm text-muted-foreground">
              Configure your primary language model provider
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <Label htmlFor="llm-name">Configuration Name *</Label>
              <Input
                id="llm-name"
                value={wizard.state.llmProvider?.name || ''}
                onChange={(e) =>
                  wizard.updateLlmProvider({
                    name: e.target.value,
                    providerType: wizard.state.llmProvider?.providerType || 'openai',
                    settingsJson: wizard.state.llmProvider?.settingsJson || "{}",
                  })
                }
                placeholder="my-openai-llm"
                className="mt-1"
              />
              <p className="text-xs text-muted-foreground mt-1">
                A unique name for this LLM configuration
              </p>
            </div>

            <div>
              <Label htmlFor="llm-provider-type">Provider Type *</Label>
              <select
                id="llm-provider-type"
                value={wizard.state.llmProvider?.providerType || (llmProviderTypes[0] || '')}
                onChange={(e) =>
                  wizard.updateLlmProvider({
                    name: wizard.state.llmProvider?.name || '',
                    providerType: e.target.value,
                    settingsJson: wizard.state.llmProvider?.settingsJson || "{}",
                  })
                }
                className="w-full mt-1 px-3 py-2 border border-input rounded-md"
                disabled={llmProviderTypes.length === 0}
              >
                {llmProviderTypes.length === 0 ? (
                  <option value="">Loading providers...</option>
                ) : (
                  llmProviderTypes.map((providerType) => (
                    <option key={providerType} value={providerType}>
                      {providerType}
                    </option>
                  ))
                )}
              </select>
              <p className="text-xs text-muted-foreground mt-1">
                Available LLM providers from the registry
              </p>
            </div>

            <Separator />

            {/* Schema-based settings form */}
            {llmSchemaLoading && (
              <div className="flex items-center justify-center py-8">
                <Spinner className="h-6 w-6" />
                <span className="ml-2 text-sm text-muted-foreground">Loading schema...</span>
              </div>
            )}

            {llmSchemaError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Failed to load provider schema: {(llmSchemaError as Error).message}
                </AlertDescription>
              </Alert>
            )}

            {llmSchema && !llmSchemaLoading && (
              <div className="space-y-4">
                <Label>Provider Settings</Label>
                {llmSchema.fields
                  .filter((f) => f.name === 'settings')
                  .map((field) => (
                    <SchemaFieldInput
                      key="llm-settings"
                      keyPrefix="wizard-llm"
                      field={field}
                      value={wizard.state.llmProvider?.settingsJson || '{}'}
                      onChange={(val) =>
                        wizard.updateLlmProvider({
                          name: wizard.state.llmProvider?.name || '',
                          providerType: wizard.state.llmProvider?.providerType || '',
                          settingsJson: String(val ?? '{}'),
                        })
                      }
                    />
                  ))}
              </div>
            )}
          </div>

          <div className="flex justify-between pt-4">
            <Button type="button" variant="outline" onClick={wizard.prevStep}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button
              type="button"
              onClick={wizard.nextStep}
              disabled={!wizard.state.llmProvider?.name || !wizard.state.llmProvider?.providerType}
            >
              Next
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </div>
      )}

      {/* Step 4: Embedding Provider */}
      {wizard.currentStep === 'embedding-provider' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Embedding Provider Configuration
            </h3>
            <p className="text-sm text-muted-foreground">
              Configure your embedding model provider for knowledge and semantic search
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <Label htmlFor="embedding-name">Configuration Name *</Label>
              <Input
                id="embedding-name"
                value={wizard.state.embeddingProvider?.name || ''}
                onChange={(e) =>
                  wizard.updateEmbeddingProvider({
                    name: e.target.value,
                    providerType: wizard.state.embeddingProvider?.providerType || 'openai',
                    settingsJson: wizard.state.embeddingProvider?.settingsJson || "{}",
                  })
                }
                placeholder="my-embedding-model"
                className="mt-1"
              />
              <p className="text-xs text-muted-foreground mt-1">
                A unique name for this embedding configuration
              </p>
            </div>

            <div>
              <Label htmlFor="embedding-provider-type">Provider Type *</Label>
              <select
                id="embedding-provider-type"
                value={wizard.state.embeddingProvider?.providerType || (embeddingProviderTypes[0] || '')}
                onChange={(e) =>
                  wizard.updateEmbeddingProvider({
                    name: wizard.state.embeddingProvider?.name || '',
                    providerType: e.target.value,
                    settingsJson: wizard.state.embeddingProvider?.settingsJson || "{}",
                  })
                }
                className="w-full mt-1 px-3 py-2 border border-input rounded-md"
                disabled={embeddingProviderTypes.length === 0}
              >
                {embeddingProviderTypes.length === 0 ? (
                  <option value="">Loading providers...</option>
                ) : (
                  embeddingProviderTypes.map((providerType) => (
                    <option key={providerType} value={providerType}>
                      {providerType}
                    </option>
                  ))
                )}
              </select>
              <p className="text-xs text-muted-foreground mt-1">
                Available embedding providers from the registry
              </p>
            </div>

            <Separator />

            {/* Schema-based settings form */}
            {embeddingSchemaLoading && (
              <div className="flex items-center justify-center py-8">
                <Spinner className="h-6 w-6" />
                <span className="ml-2 text-sm text-muted-foreground">Loading schema...</span>
              </div>
            )}

            {embeddingSchemaError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Failed to load provider schema: {(embeddingSchemaError as Error).message}
                </AlertDescription>
              </Alert>
            )}

            {embeddingSchema && !embeddingSchemaLoading && (
              <div className="space-y-4">
                <Label>Provider Settings</Label>
                {embeddingSchema.fields
                  .filter((f) => f.name === 'settings')
                  .map((field) => (
                    <SchemaFieldInput
                      key="embedding-settings"
                      keyPrefix="wizard-embedding"
                      field={field}
                      value={wizard.state.embeddingProvider?.settingsJson || '{}'}
                      onChange={(val) =>
                        wizard.updateEmbeddingProvider({
                          name: wizard.state.embeddingProvider?.name || '',
                          providerType: wizard.state.embeddingProvider?.providerType || '',
                          settingsJson: String(val ?? '{}'),
                        })
                      }
                    />
                  ))}
              </div>
            )}
          </div>

          <div className="flex justify-between pt-4">
            <Button type="button" variant="outline" onClick={wizard.prevStep}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button
              type="button"
              onClick={wizard.nextStep}
              disabled={!wizard.state.embeddingProvider?.name || !wizard.state.embeddingProvider?.providerType}
            >
              Next
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </div>
      )}

      {/* Step 5: Vector & Graph Databases */}
      {wizard.currentStep === 'vector-db' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
              <Database className="h-5 w-5" />
              Database Configuration
            </h3>
            <p className="text-sm text-muted-foreground">
              Configure vector and graph databases for knowledge management (optional)
            </p>
          </div>

          <div className="space-y-4">
            <div className="p-4 border-2 border-dashed rounded-lg">
              <h4 className="font-medium mb-2">Vector Database (Optional)</h4>
              <p className="text-sm text-muted-foreground mb-3">
                For semantic search and retrieval augmented generation
              </p>

              <div className="space-y-3">
                <div>
                  <Label htmlFor="vector-db-name">Configuration Name</Label>
                  <Input
                    id="vector-db-name"
                    value={wizard.state.vectorDb?.name || ''}
                    onChange={(e) =>
                      wizard.updateVectorDb({
                        name: e.target.value,
                        providerType: wizard.state.vectorDb?.providerType || 'qdrant',
                        settingsJson: wizard.state.vectorDb?.settingsJson || "{}",
                      })
                    }
                    placeholder="my-vector-db"
                    className="mt-1"
                  />
                </div>

                <div>
                  <Label htmlFor="vector-db-type">Provider Type</Label>
                  <select
                    id="vector-db-type"
                    value={wizard.state.vectorDb?.providerType || (vectorDbProviderTypes[0] || '')}
                    onChange={(e) =>
                      wizard.updateVectorDb({
                        name: wizard.state.vectorDb?.name || '',
                        providerType: e.target.value,
                        settingsJson: wizard.state.vectorDb?.settingsJson || "{}",
                      })
                    }
                    className="w-full mt-1 px-3 py-2 border border-input rounded-md"
                    disabled={!wizard.state.vectorDb?.name || vectorDbProviderTypes.length === 0}
                  >
                    {vectorDbProviderTypes.length === 0 ? (
                      <option value="">Loading providers...</option>
                    ) : (
                      vectorDbProviderTypes.map((providerType) => (
                        <option key={providerType} value={providerType}>
                          {providerType}
                        </option>
                      ))
                    )}
                  </select>
                </div>

                {/* Vector DB Schema-based settings */}
                {wizard.state.vectorDb?.name && wizard.state.vectorDb?.providerType && (
                  <>
                    {vectorDbSchemaLoading && (
                      <div className="flex items-center py-4">
                        <Spinner className="h-4 w-4" />
                        <span className="ml-2 text-sm text-muted-foreground">Loading schema...</span>
                      </div>
                    )}

                    {vectorDbSchemaError && (
                      <Alert variant="destructive" className="mt-2">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>
                          Failed to load schema: {(vectorDbSchemaError as Error).message}
                        </AlertDescription>
                      </Alert>
                    )}

                    {vectorDbSchema && !vectorDbSchemaLoading && (
                      <div className="space-y-3 mt-3">
                        <Label className="text-xs">Database Settings</Label>
                        {vectorDbSchema.fields
                          .filter((f) => f.name === 'settings')
                          .map((field) => (
                            <SchemaFieldInput
                              key="vector-db-settings"
                              keyPrefix="wizard-vector-db"
                              field={field}
                              value={wizard.state.vectorDb?.settingsJson || '{}'}
                              onChange={(val) =>
                                wizard.updateVectorDb({
                                  name: wizard.state.vectorDb?.name || '',
                                  providerType: wizard.state.vectorDb?.providerType || '',
                                  settingsJson: String(val ?? '{}'),
                                })
                              }
                            />
                          ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            <div className="p-4 border-2 border-dashed rounded-lg">
              <h4 className="font-medium mb-2">Graph Database (Optional)</h4>
              <p className="text-sm text-muted-foreground mb-3">
                For relationship-based knowledge queries
              </p>

              <div className="space-y-3">
                <div>
                  <Label htmlFor="graph-db-name">Configuration Name</Label>
                  <Input
                    id="graph-db-name"
                    value={wizard.state.graphDb?.name || ''}
                    onChange={(e) =>
                      wizard.updateGraphDb({
                        name: e.target.value,
                        providerType: wizard.state.graphDb?.providerType || 'neo4j',
                        settingsJson: wizard.state.graphDb?.settingsJson || "{}",
                      })
                    }
                    placeholder="my-graph-db"
                    className="mt-1"
                  />
                </div>

                <div>
                  <Label htmlFor="graph-db-type">Provider Type</Label>
                  <select
                    id="graph-db-type"
                    value={wizard.state.graphDb?.providerType || (graphDbProviderTypes[0] || '')}
                    onChange={(e) =>
                      wizard.updateGraphDb({
                        name: wizard.state.graphDb?.name || '',
                        providerType: e.target.value,
                        settingsJson: wizard.state.graphDb?.settingsJson || "{}",
                      })
                    }
                    className="w-full mt-1 px-3 py-2 border border-input rounded-md"
                    disabled={!wizard.state.graphDb?.name || graphDbProviderTypes.length === 0}
                  >
                    {graphDbProviderTypes.length === 0 ? (
                      <option value="">Loading providers...</option>
                    ) : (
                      graphDbProviderTypes.map((providerType) => (
                        <option key={providerType} value={providerType}>
                          {providerType}
                        </option>
                      ))
                    )}
                  </select>
                </div>

                {/* Graph DB Schema-based settings */}
                {wizard.state.graphDb?.name && wizard.state.graphDb?.providerType && (
                  <>
                    {graphDbSchemaLoading && (
                      <div className="flex items-center py-4">
                        <Spinner className="h-4 w-4" />
                        <span className="ml-2 text-sm text-muted-foreground">Loading schema...</span>
                      </div>
                    )}

                    {graphDbSchemaError && (
                      <Alert variant="destructive" className="mt-2">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>
                          Failed to load schema: {(graphDbSchemaError as Error).message}
                        </AlertDescription>
                      </Alert>
                    )}

                    {graphDbSchema && !graphDbSchemaLoading && (
                      <div className="space-y-3 mt-3">
                        <Label className="text-xs">Database Settings</Label>
                        {graphDbSchema.fields
                          .filter((f) => f.name === 'settings')
                          .map((field) => (
                            <SchemaFieldInput
                              key="graph-db-settings"
                              keyPrefix="wizard-graph-db"
                              field={field}
                              value={wizard.state.graphDb?.settingsJson || '{}'}
                              onChange={(val) =>
                                wizard.updateGraphDb({
                                  name: wizard.state.graphDb?.name || '',
                                  providerType: wizard.state.graphDb?.providerType || '',
                                  settingsJson: String(val ?? '{}'),
                                })
                              }
                            />
                          ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            <div className="p-4 bg-muted/50 rounded-lg">
              <p className="text-sm text-muted-foreground">
                Database configurations are optional. You can skip this step and add them later on the Configs page.
              </p>
            </div>
          </div>

          <div className="flex justify-between pt-4">
            <Button type="button" variant="outline" onClick={wizard.prevStep}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button type="button" onClick={wizard.nextStep}>
              Next
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </div>
      )}

      {/* Step 6: Model Configuration */}
      {wizard.currentStep === 'model-config' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Model Configuration
            </h3>
            <p className="text-sm text-muted-foreground">
              Create a model configuration that references your LLM provider (optional)
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <Label htmlFor="model-name">Configuration Name</Label>
              <Input
                id="model-name"
                value={wizard.state.modelConfig?.name || ''}
                onChange={(e) =>
                  wizard.updateModelConfig({
                    name: e.target.value,
                    providerType: wizard.state.llmProvider?.providerType || llmProviderTypes[0] || '',
                    configJson: wizard.state.modelConfig?.configJson || "{}",
                  })
                }
                placeholder="my-model-config"
                className="mt-1"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Optional: Create a named model configuration for reuse across agents
              </p>
            </div>

            <Separator />

            {/* Schema-based config form */}
            {wizard.state.modelConfig?.name && (
              <>
                {modelConfigSchemaLoading && (
                  <div className="flex items-center justify-center py-8">
                    <Spinner className="h-6 w-6" />
                    <span className="ml-2 text-sm text-muted-foreground">Loading schema...</span>
                  </div>
                )}

                {modelConfigSchemaError && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      Failed to load model config schema: {(modelConfigSchemaError as Error).message}
                    </AlertDescription>
                  </Alert>
                )}

                {modelConfigSchema && !modelConfigSchemaLoading && (
                  <div className="space-y-4">
                    <Label>Model Configuration</Label>
                    {modelConfigSchema.fields
                      .filter((f) => f.name === 'config')
                      .map((field) => (
                        <SchemaFieldInput
                          key="model-config"
                          keyPrefix="wizard-model"
                          field={field}
                          value={wizard.state.modelConfig?.configJson || '{}'}
                          onChange={(val) =>
                            wizard.updateModelConfig({
                              name: wizard.state.modelConfig?.name || '',
                              providerType: wizard.state.modelConfig?.providerType || '',
                              configJson: String(val ?? '{}'),
                            })
                          }
                        />
                      ))}
                  </div>
                )}
              </>
            )}

            {!wizard.state.modelConfig?.name && (
              <div className="p-4 bg-muted/50 rounded-lg">
                <p className="text-sm text-muted-foreground">
                  Model configurations let you define reusable settings (temperature, max tokens, etc.)
                  that can be shared across multiple agents. Enter a name above to configure, or skip
                  this step to add it later on the Configs page.
                </p>
              </div>
            )}
          </div>

          <div className="flex justify-between pt-4">
            <Button type="button" variant="outline" onClick={wizard.prevStep}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button type="button" onClick={wizard.nextStep}>
              Next
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </div>
      )}

      {/* Step 7: Agents & Tools */}
      {wizard.currentStep === 'agents-tools' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2">Agent Configuration</h3>
            <p className="text-sm text-muted-foreground">
              Configure the default agent for your project
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <Label htmlFor="agent-name">Agent Name</Label>
              <Input
                id="agent-name"
                value={wizard.state.agentConfig?.name || 'default-agent'}
                onChange={(e) => wizard.updateAgentConfig({
                  name: e.target.value,
                  configJson: wizard.state.agentConfig?.configJson || '{}'
                })}
                placeholder="default-agent"
                className="mt-1"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Name for the agent configuration file
              </p>
            </div>

            <Separator />

            {/* Schema-based agent config form */}
            {agentSchemaLoading && (
              <div className="flex items-center justify-center py-8">
                <Spinner className="h-6 w-6" />
                <span className="ml-2 text-sm">Loading agent schema...</span>
              </div>
            )}

            {agentSchemaError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Failed to load agent schema: {(agentSchemaError as Error).message}
                </AlertDescription>
              </Alert>
            )}

            {agentSchema && !agentSchemaLoading && wizard.state.agentConfig && (
              <div className="space-y-4">
                <Label>Agent Settings</Label>
                {agentSchema.fields.map((field) => (
                  <SchemaFieldInput
                    key={field.name}
                    keyPrefix="wizard-agent"
                    field={field}
                    value={(() => {
                      try {
                        const config = JSON.parse(wizard.state.agentConfig?.configJson || '{}')
                        return config[field.name] ?? field.default
                      } catch {
                        return field.default
                      }
                    })()}
                    onChange={(val) => {
                      try {
                        const config = JSON.parse(wizard.state.agentConfig?.configJson || '{}')
                        config[field.name] = val
                        wizard.updateAgentConfig({
                          name: wizard.state.agentConfig?.name || 'default-agent',
                          configJson: JSON.stringify(config),
                        })
                      } catch (error) {
                        console.error('Failed to update agent config:', error)
                      }
                    }}
                  />
                ))}
              </div>
            )}
          </div>

          <div className="flex justify-between pt-4">
            <Button type="button" variant="outline" onClick={wizard.prevStep}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button type="button" onClick={wizard.nextStep}>
              Next
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </div>
      )}

      {/* Step 8: Review */}
      {wizard.currentStep === 'review' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2">Review and Create</h3>
            <p className="text-sm text-muted-foreground">
              Review your project configuration before creating
            </p>
          </div>

          <div className="space-y-2 p-4 bg-muted rounded-lg">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Project Name:</span>
              <span className="font-medium">{wizard.state.project.name}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Description:</span>
              <span className="font-medium">
                {wizard.state.project.description || 'No description'}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Setup Type:</span>
              <span className="font-medium capitalize">{wizard.state.project.setupType}</span>
            </div>

            {wizard.state.project.setupType === 'fast' && wizard.state.fastConfig && (
              <>
                <Separator className="my-2" />
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">LLM Provider:</span>
                  <span className="font-medium">{wizard.state.fastConfig.llmProviderType}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Model Path:</span>
                  <span className="font-medium truncate max-w-xs">
                    {wizard.state.fastConfig.modelPath}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Configurations:</span>
                  <span className="font-medium">All defaults will be created</span>
                </div>
              </>
            )}

            {wizard.state.project.setupType === 'guided' && (
              <>
                <Separator className="my-2" />
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">LLM Provider:</span>
                  <span className="font-medium">
                    {wizard.state.llmProvider
                      ? `${wizard.state.llmProvider.name} (${wizard.state.llmProvider.providerType})`
                      : 'Not configured'}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Embedding Provider:</span>
                  <span className="font-medium">
                    {wizard.state.embeddingProvider
                      ? `${wizard.state.embeddingProvider.name} (${wizard.state.embeddingProvider.providerType})`
                      : 'Not configured'}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Vector DB:</span>
                  <span className="font-medium">
                    {wizard.state.vectorDb
                      ? `${wizard.state.vectorDb.name} (${wizard.state.vectorDb.providerType})`
                      : 'None'}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Graph DB:</span>
                  <span className="font-medium">
                    {wizard.state.graphDb
                      ? `${wizard.state.graphDb.name} (${wizard.state.graphDb.providerType})`
                      : 'None'}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Model Config:</span>
                  <span className="font-medium">
                    {wizard.state.modelConfig ? wizard.state.modelConfig.name : 'None'}
                  </span>
                </div>
                <Separator className="my-2" />
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Agents:</span>
                  <span className="font-medium">
                    {wizard.state.project.agentNames || 'None'}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Tool Categories:</span>
                  <span className="font-medium">
                    {wizard.state.project.toolCategories || 'None'}
                  </span>
                </div>
              </>
            )}
          </div>

          <div className="flex justify-between pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={wizard.prevStep}
              disabled={wizard.isCreating}
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button
              type="button"
              onClick={() => wizard.createProject(onComplete)}
              disabled={wizard.isCreating}
            >
              {wizard.isCreating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  <FolderOpen className="h-4 w-4 mr-2" />
                  Create Project
                </>
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

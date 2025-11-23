import { useState, useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { RenameDeleteControls } from '../RenameDeleteControls'
import { DiffPreview } from './DiffPreview'
import { Button } from '../../ui/Button'
import { Input } from '../../ui/Input'
import { Textarea } from '../../ui/Textarea'
import { Label } from '../../ui/Label'
import { KnowledgePluginMultiSelect } from '../../knowledge/KnowledgePluginMultiSelect'
import type { EditorTarget } from './ConfigEditor'
import type { UseConfigMutationsResult } from '../../../hooks/configs/useConfigMutations'
import type { AgentConfigSummary } from '../../../services/configs'

export interface AgentEditorFormProps {
  projectId: string
  target: EditorTarget
  mutations: UseConfigMutationsResult
  onSuccess: (message: string) => void
  onError: (message: string) => void
  onDiff: () => void
  onApply: () => void
}

/**
 * Structured editor form for agent configurations.
 */
export function AgentEditorForm({
  projectId,
  target,
  mutations,
  onSuccess,
  onError,
  onDiff,
  onApply,
}: AgentEditorFormProps) {
  const queryClient = useQueryClient()
  const agentData = target.data as AgentConfigSummary

  const [name, setName] = useState(agentData?.name || target.name)
  const [persona, setPersona] = useState(agentData?.persona || '')
  const [allowedToolCategories, setAllowedToolCategories] = useState(
    agentData?.allowed_tool_categories?.join('\n') || ''
  )
  const [knowledgePlugins, setKnowledgePlugins] = useState<string[]>(agentData?.knowledge_plugins || [])
  const [modelName, setModelName] = useState(agentData?.model_name || '')
  const [llmName, setLlmName] = useState(agentData?.llm_name || '')
  const [temperature, setTemperature] = useState(String(agentData?.temperature ?? '0.7'))
  const [maxIterations, setMaxIterations] = useState(String(agentData?.max_iterations ?? '10'))
  const [enableLearning, setEnableLearning] = useState(agentData?.enable_learning ?? false)
  const [verbose, setVerbose] = useState(agentData?.verbose ?? false)

  // Reset form when target changes
  useEffect(() => {
    if (target.type === 'agent' && target.data) {
      const data = target.data as AgentConfigSummary
      setName(data.name)
      setPersona(data.persona || '')
      setAllowedToolCategories(data.allowed_tool_categories?.join('\n') || '')
      setKnowledgePlugins(data.knowledge_plugins || [])
      setModelName(data.model_name || '')
      setLlmName(data.llm_name || '')
      setTemperature(String(data.temperature ?? '0.7'))
      setMaxIterations(String(data.max_iterations ?? '10'))
      setEnableLearning(data.enable_learning ?? false)
      setVerbose(data.verbose ?? false)
    }
  }, [target])

  return (
    <form className="grid gap-4 max-w-2xl">
      <div className="grid gap-2">
        <Label htmlFor="agent-name">Name</Label>
        <Input
          id="agent-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        {mutations.structuredErrors['agent.name'] && (
          <span className="text-sm text-destructive">{mutations.structuredErrors['agent.name']}</span>
        )}
      </div>

      <div className="grid gap-2">
        <Label htmlFor="agent-persona">Persona</Label>
        <Textarea
          id="agent-persona"
          rows={4}
          value={persona}
          onChange={(e) => setPersona(e.target.value)}
          placeholder="Describe the agent's role and behavior..."
        />
        {mutations.structuredErrors['agent.persona'] && (
          <span className="text-sm text-destructive">{mutations.structuredErrors['agent.persona']}</span>
        )}
      </div>

      <div className="grid gap-2">
        <Label htmlFor="agent-tool-categories">Allowed Tool Categories</Label>
        <Textarea
          id="agent-tool-categories"
          rows={4}
          value={allowedToolCategories}
          onChange={(e) => setAllowedToolCategories(e.target.value)}
          placeholder="One category per line (e.g., file_operations, web_search)"
          className="font-mono"
        />
        <p className="text-sm text-muted-foreground">One category per line</p>
        {mutations.structuredErrors['agent.allowed_tool_categories'] && (
          <span className="text-sm text-destructive">
            {mutations.structuredErrors['agent.allowed_tool_categories']}
          </span>
        )}
      </div>

      <KnowledgePluginMultiSelect
        projectId={projectId}
        selectedPluginIds={knowledgePlugins}
        onChange={setKnowledgePlugins}
        error={mutations.structuredErrors['agent.knowledge_plugins']}
      />

      <div className="grid gap-2">
        <Label htmlFor="agent-model-name">Model Name</Label>
        <Input
          id="agent-model-name"
          type="text"
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          placeholder="e.g., gpt-4, claude-3-opus"
        />
        {mutations.structuredErrors['agent.model_name'] && (
          <span className="text-sm text-destructive">{mutations.structuredErrors['agent.model_name']}</span>
        )}
      </div>

      <div className="grid gap-2">
        <Label htmlFor="agent-llm-name">LLM Name</Label>
        <Input
          id="agent-llm-name"
          type="text"
          value={llmName}
          onChange={(e) => setLlmName(e.target.value)}
          placeholder="LLM configuration name"
        />
        {mutations.structuredErrors['agent.llm_name'] && (
          <span className="text-sm text-destructive">{mutations.structuredErrors['agent.llm_name']}</span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="grid gap-2">
          <Label htmlFor="agent-temperature">Temperature</Label>
          <Input
            id="agent-temperature"
            type="number"
            step="0.1"
            min="0"
            max="2"
            value={temperature}
            onChange={(e) => setTemperature(e.target.value)}
          />
          {mutations.structuredErrors['agent.temperature'] && (
            <span className="text-sm text-destructive">{mutations.structuredErrors['agent.temperature']}</span>
          )}
        </div>

        <div className="grid gap-2">
          <Label htmlFor="agent-max-iterations">Max Iterations</Label>
          <Input
            id="agent-max-iterations"
            type="number"
            min="1"
            value={maxIterations}
            onChange={(e) => setMaxIterations(e.target.value)}
          />
          {mutations.structuredErrors['agent.max_iterations'] && (
            <span className="text-sm text-destructive">{mutations.structuredErrors['agent.max_iterations']}</span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="flex items-center gap-2">
          <input
            id="agent-enable-learning"
            type="checkbox"
            checked={enableLearning}
            onChange={(e) => setEnableLearning(e.target.checked)}
            className="w-4 h-4"
          />
          <Label htmlFor="agent-enable-learning" className="cursor-pointer">
            Enable Learning
          </Label>
        </div>

        <div className="flex items-center gap-2">
          <input
            id="agent-verbose"
            type="checkbox"
            checked={verbose}
            onChange={(e) => setVerbose(e.target.checked)}
            className="w-4 h-4"
          />
          <Label htmlFor="agent-verbose" className="cursor-pointer">
            Verbose
          </Label>
        </div>
      </div>

      {/* Rename/Delete actions */}
      <fieldset className="border border-border rounded-md p-4">
        <legend className="px-2 text-sm font-medium">File Operations</legend>
        <div className="grid gap-3">
          <div className="grid gap-2">
            <Label htmlFor="agent-current-path">Current Path</Label>
            <Input
              id="agent-current-path"
              type="text"
              value={target.relativePath}
              readOnly
              className="bg-muted cursor-not-allowed"
            />
          </div>
          <RenameDeleteControls
            projectId={projectId}
            currentPath={target.relativePath}
            onRenamed={() => {
              onSuccess('Renamed successfully.')
              queryClient.invalidateQueries({ queryKey: ['configs'] })
            }}
            onDeleted={() => {
              onSuccess('Deleted successfully.')
              queryClient.invalidateQueries({ queryKey: ['configs'] })
            }}
            onError={onError}
          />
        </div>
      </fieldset>

      <div className="flex gap-3">
        <Button type="button" onClick={onDiff} disabled={mutations.computeStructuredDiff.isPending}>
          {mutations.computeStructuredDiff.isPending ? 'Diffing…' : 'Show Diff'}
        </Button>
        <Button type="button" onClick={onApply} disabled={mutations.applyStructuredMutation.isPending} variant="secondary">
          {mutations.applyStructuredMutation.isPending ? 'Applying…' : 'Apply'}
        </Button>
      </div>

      <DiffPreview diffResult={mutations.structuredDiff} />
    </form>
  )
}

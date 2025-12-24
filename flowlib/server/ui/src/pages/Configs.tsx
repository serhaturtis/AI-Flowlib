import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { Settings, Eye, AlertCircle, Plus, CheckCircle2 } from 'lucide-react'

import { useProjectContext } from '../contexts/ProjectContext'
import { useUrlState } from '../hooks/useUrlState'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { Alert, AlertDescription } from '../components/ui/Alert'
import { Stack } from '../components/layout/Stack'
import { SplitPane } from '../components/layout/SplitPane'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/Tabs'
import {
  fetchAgentConfig,
  fetchProviderConfigs,
  fetchResourceConfigs,
  fetchMessageSources,
  AgentConfigSummary,
  ProviderConfigSummary,
  ResourceConfigSummary,
  MessageSourceSummary,
  ConfigDiffResponse,
} from '../services/configs'
import { ConfigEditor, type EditorTarget } from '../components/configs/ConfigEditor'
import { AgentConfigList } from '../components/configs/ConfigLists/AgentConfigList'
import { ProviderConfigList } from '../components/configs/ConfigLists/ProviderConfigList'
import { ResourceConfigList } from '../components/configs/ConfigLists/ResourceConfigList'
import { MessageSourceConfigList } from '../components/configs/ConfigLists/MessageSourceConfigList'
import { AliasesTab } from '../components/configs/ConfigLists/AliasesTab'
import { CreateConfigDialog } from '../components/configs/CreateConfigDialog'
import { useConfigQueries } from '../hooks/configs/useConfigQueries'
import { resolveResourcePath } from '../utils/configs/configHelpers'

// Constants for Fast Refresh compliance
const VALID_TAB_VALUES = ['agents', 'providers', 'resources', 'sources', 'aliases'] as const

type DetailState = {
  agent?: AgentConfigSummary | null
  provider?: ProviderConfigSummary | null
  resource?: ResourceConfigSummary | null
  messageSource?: MessageSourceSummary | null
}

export default function Configs() {
  const queryClient = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()
  const [details, setDetails] = useState<DetailState>({})
  const [detailError, setDetailError] = useState<string | null>(null)
  const [editorTarget, setEditorTarget] = useState<EditorTarget | null>(null)
  const [editorContent, setEditorContent] = useState('')
  const [editorBaseHash, setEditorBaseHash] = useState('')
  const [diffResult, setDiffResult] = useState<ConfigDiffResponse | null>(null)
  const [editorError, setEditorError] = useState<string | null>(null)
  const [editorSuccess, setEditorSuccess] = useState<string | null>(null)
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [createDialogType, setCreateDialogType] = useState<'provider' | 'resource' | 'message_source'>('provider')

  // Use global project context
  const { selectedProjectId: selectedProject, selectedProject: selectedProjectMeta, projects } =
    useProjectContext()

  // Fetch all configs using centralized hook
  const { agentsQuery, providersQuery, resourcesQuery, aliasesQuery, messageSourcesQuery } = useConfigQueries(selectedProject)

  // Sync active tab with URL param
  const [activeTab, setActiveTab] = useUrlState<'agents' | 'providers' | 'resources' | 'sources' | 'aliases'>(
    'tab',
    'agents',
    VALID_TAB_VALUES as unknown as ('agents' | 'providers' | 'resources' | 'sources' | 'aliases')[],
  )

  // Handle URL-based actions
  useEffect(() => {
    const openParam = searchParams.get('open')
    if (!openParam) return

    const newParams = new URLSearchParams(searchParams)

    if (openParam === 'provider-create') {
      setCreateDialogType('provider')
      setCreateDialogOpen(true)
    } else if (openParam === 'resource-create') {
      setCreateDialogType('resource')
      setCreateDialogOpen(true)
    } else if (openParam === 'message-source-create') {
      setCreateDialogType('message_source')
      setCreateDialogOpen(true)
    } else if (openParam === 'aliases') {
      setActiveTab('aliases')
    }

    newParams.delete('open')
    setSearchParams(newParams, { replace: true })
  }, [searchParams, setActiveTab, setSearchParams])

  // Sync selected config with URL param (simplified - handlers will update URL)
  useEffect(() => {
    const configParam = searchParams.get('config')
    const configTypeParam = searchParams.get('configType')
    if (configParam && configTypeParam && selectedProject) {
      // Only sync if not already selected (avoid loops)
      const currentConfig =
        (configTypeParam === 'agent' && details.agent?.name === configParam) ||
        (configTypeParam === 'provider' && details.provider?.name === configParam) ||
        (configTypeParam === 'resource' && details.resource?.name === configParam) ||
        (configTypeParam === 'messageSource' && details.messageSource?.name === configParam)

      if (currentConfig) {
        return // Already selected, no need to reload
      }

      // Load config based on URL params
      if (configTypeParam === 'agent') {
        fetchAgentConfig(selectedProject, configParam)
          .then((data) => {
            setDetails((prev) => ({ ...prev, agent: data }))
            setDetailError(null)
          })
          .catch((error) => {
            // Config not found, remove from URL and show error
            const message = error instanceof Error ? error.message : `Failed to load agent config: ${configParam}`
            setDetailError(message)
            const newParams = new URLSearchParams(searchParams)
            newParams.delete('config')
            newParams.delete('configType')
            setSearchParams(newParams, { replace: true })
          })
      } else if (configTypeParam === 'provider') {
        fetchProviderConfigs(selectedProject)
          .then((response) => {
            const config = response.configs.find((c) => c.name === configParam)
            if (config) {
              setDetails((prev) => ({ ...prev, provider: config }))
              setEditorTarget({
                type: 'provider',
                name: config.name,
                relativePath: `configs/providers/${config.name}.py`,
                resourceType: config.resource_type,
                providerType: config.provider_type,
                data: config.settings,
              })
              setEditorContent(JSON.stringify(config.settings, null, 2))
              setEditorBaseHash('')
              setDiffResult(null)
              setEditorError(null)
              setEditorSuccess(null)
              setDetailError(null)
            } else {
              setDetailError(`Provider config not found: ${configParam}`)
              const newParams = new URLSearchParams(searchParams)
              newParams.delete('config')
              newParams.delete('configType')
              setSearchParams(newParams, { replace: true })
            }
          })
          .catch((error) => {
            const message = error instanceof Error ? error.message : `Failed to load provider configs`
            setDetailError(message)
            const newParams = new URLSearchParams(searchParams)
            newParams.delete('config')
            newParams.delete('configType')
            setSearchParams(newParams, { replace: true })
          })
      } else if (configTypeParam === 'resource') {
        fetchResourceConfigs(selectedProject)
          .then((response) => {
            const config = response.configs.find((c) => c.name === configParam)
            if (config) {
              setDetails((prev) => ({ ...prev, resource: config }))
              setEditorTarget({
                type: 'resource',
                name: config.name,
                relativePath: resolveResourcePath(config),
                resourceType: config.resource_type,
                providerType: (config.metadata?.provider_type as string) ?? '',
                data: (config.metadata?.config as Record<string, unknown>) ?? {},
              })
              setEditorContent(JSON.stringify(config.metadata, null, 2))
              setEditorBaseHash('')
              setDiffResult(null)
              setEditorError(null)
              setEditorSuccess(null)
              setDetailError(null)
            } else {
              setDetailError(`Resource config not found: ${configParam}`)
              const newParams = new URLSearchParams(searchParams)
              newParams.delete('config')
              newParams.delete('configType')
              setSearchParams(newParams, { replace: true })
            }
          })
          .catch((error) => {
            const message = error instanceof Error ? error.message : `Failed to load resource configs`
            setDetailError(message)
            const newParams = new URLSearchParams(searchParams)
            newParams.delete('config')
            newParams.delete('configType')
            setSearchParams(newParams, { replace: true })
          })
      } else if (configTypeParam === 'messageSource') {
        fetchMessageSources(selectedProject)
          .then((response) => {
            const config = response.sources.find((c) => c.name === configParam)
            if (config) {
              setDetails((prev) => ({ ...prev, messageSource: config }))
              setEditorTarget({
                type: 'message_source',
                name: config.name,
                relativePath: `configs/message_sources/${config.name}.py`,
                sourceType: config.source_type,
                data: { ...config.settings, enabled: config.enabled },
              })
              setEditorContent(JSON.stringify(config.settings, null, 2))
              setEditorBaseHash('')
              setDiffResult(null)
              setEditorError(null)
              setEditorSuccess(null)
              setDetailError(null)
            } else {
              setDetailError(`Message source config not found: ${configParam}`)
              const newParams = new URLSearchParams(searchParams)
              newParams.delete('config')
              newParams.delete('configType')
              setSearchParams(newParams, { replace: true })
            }
          })
          .catch((error) => {
            const message = error instanceof Error ? error.message : `Failed to load message source configs`
            setDetailError(message)
            const newParams = new URLSearchParams(searchParams)
            newParams.delete('config')
            newParams.delete('configType')
            setSearchParams(newParams, { replace: true })
          })
      }
    }
  }, [searchParams, selectedProject, setSearchParams, details])


  const handleSelectAgent = async (name: string) => {
    try {
      setDetailError(null)
      const data = await fetchAgentConfig(selectedProject, name)
      setDetails({ provider: null, resource: null, agent: data, messageSource: null })
      setEditorTarget({
        type: 'agent',
        name: data.name,
        relativePath: `configs/agents/${data.name}.py`,
        data: data,
      })
      setEditorContent(JSON.stringify(data, null, 2))
      setEditorBaseHash('')
      setDiffResult(null)
      setEditorError(null)
      setEditorSuccess(null)
      // Update URL param
      const newParams = new URLSearchParams(searchParams)
      newParams.set('config', name)
      newParams.set('configType', 'agent')
      setSearchParams(newParams, { replace: true })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load agent config.'
      setDetailError(message)
    }
  }

  const handleSelectProvider = (config: ProviderConfigSummary) => {
    setDetails({ provider: config, resource: null, agent: null, messageSource: null })
    setEditorTarget({
      type: 'provider',
      name: config.name,
      relativePath: `configs/providers/${config.name}.py`,
      resourceType: config.resource_type,
      providerType: config.provider_type,
      data: config.settings,
    })
    setEditorContent(JSON.stringify(config.settings, null, 2))
    setEditorBaseHash('')
    setDiffResult(null)
    setEditorError(null)
    setEditorSuccess(null)
    // Update URL param
    const newParams = new URLSearchParams(searchParams)
    newParams.set('config', config.name)
    newParams.set('configType', 'provider')
    setSearchParams(newParams, { replace: true })
  }

  const handleSelectResource = (config: ResourceConfigSummary) => {
    setDetails({ provider: null, resource: config, agent: null, messageSource: null })
    setEditorTarget({
      type: 'resource',
      name: config.name,
      relativePath: resolveResourcePath(config),
      resourceType: config.resource_type,
      providerType: (config.metadata?.provider_type as string) ?? '',
      data: (config.metadata?.config as Record<string, unknown>) ?? {},
    })
    setEditorContent(JSON.stringify(config.metadata, null, 2))
    setEditorBaseHash('')
    setDiffResult(null)
    setEditorError(null)
    setEditorSuccess(null)
    // Update URL param
    const newParams = new URLSearchParams(searchParams)
    newParams.set('config', config.name)
    newParams.set('configType', 'resource')
    setSearchParams(newParams, { replace: true })
  }

  const handleSelectSource = (config: MessageSourceSummary) => {
    setDetails({ provider: null, resource: null, agent: null, messageSource: config })
    setEditorTarget({
      type: 'message_source',
      name: config.name,
      relativePath: `configs/message_sources/${config.name}.py`,
      sourceType: config.source_type,
      data: { ...config.settings, enabled: config.enabled },
    })
    setEditorContent(JSON.stringify(config.settings, null, 2))
    setEditorBaseHash('')
    setDiffResult(null)
    setEditorError(null)
    setEditorSuccess(null)
    // Update URL param
    const newParams = new URLSearchParams(searchParams)
    newParams.set('config', config.name)
    newParams.set('configType', 'messageSource')
    setSearchParams(newParams, { replace: true })
  }

  if (!selectedProject && projects.length > 0) {
    return (
      <Stack spacing="xl">
        <div>
          <h1 className="text-4xl font-bold tracking-tight flex items-center gap-2">
            <Settings className="h-8 w-8" />
            Configurations
          </h1>
          <p className="text-muted-foreground mt-2">
            Manage agent, provider, and resource configurations
          </p>
        </div>
        <Card>
          <CardHeader>
            <CardTitle>No Project Selected</CardTitle>
            <CardDescription>Select a project from the top bar to manage its configurations</CardDescription>
          </CardHeader>
        </Card>
      </Stack>
    )
  }

  return (
    <div className="flex flex-col -mx-4 -my-8" style={{ height: 'calc(100vh - 4rem)' }}>
      {/* Header with Project Selector - Sticky */}
      <div className="sticky top-0 z-10 border-b border-border bg-card/95 backdrop-blur-sm p-4 px-4 sm:px-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="min-w-0 flex-1">
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight flex items-center gap-2">
              <Settings className="h-5 w-5 sm:h-6 sm:w-6" />
              Configurations
            </h1>
            {selectedProjectMeta && (
              <p className="text-xs sm:text-sm text-muted-foreground mt-1 truncate">
                <span className="font-medium">Project:</span> {selectedProjectMeta.name} •{' '}
                <code className="bg-muted px-1.5 py-0.5 rounded text-xs">{selectedProjectMeta.path}</code>
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Split Pane Layout */}
      <SplitPane
        leftWidth="400px"
        rightMinWidth="600px"
        className="flex-1 min-h-0"
        left={
          <div className="h-full flex flex-col border-r border-border bg-muted/20">
            <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)} className="flex-1 flex flex-col min-h-0">
              <div className="p-2 sm:p-4 border-b border-border">
                <TabsList className="grid w-full grid-cols-5 gap-1">
                  <TabsTrigger value="agents" className="text-xs sm:text-sm">Agents</TabsTrigger>
                  <TabsTrigger value="providers" className="text-xs sm:text-sm">Providers</TabsTrigger>
                  <TabsTrigger value="resources" className="text-xs sm:text-sm">Resources</TabsTrigger>
                  <TabsTrigger value="sources" className="text-xs sm:text-sm">Sources</TabsTrigger>
                  <TabsTrigger value="aliases" className="text-xs sm:text-sm">Aliases</TabsTrigger>
                </TabsList>
              </div>
              <div className="flex-1 overflow-y-auto">
                <TabsContent value="agents" className="mt-0 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-semibold">Agent Configurations</h2>
                  </div>
                  <AgentConfigList agentsQuery={agentsQuery} onSelectAgent={handleSelectAgent} />
                </TabsContent>
                <TabsContent value="providers" className="mt-0 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-semibold">Provider Configurations</h2>
                    <Button
                      size="sm"
                      disabled={!selectedProject}
                      onClick={() => {
                        setCreateDialogType('provider')
                        setCreateDialogOpen(true)
                      }}
                    >
                      <Plus className="h-4 w-4 mr-1" />
                      Create
                    </Button>
                  </div>
                  <ProviderConfigList
                    providersQuery={providersQuery}
                    onSelectProvider={handleSelectProvider}
                    onCreateProvider={() => {
                      setCreateDialogType('provider')
                      setCreateDialogOpen(true)
                    }}
                    selectedProject={selectedProject}
                  />
                </TabsContent>
                <TabsContent value="resources" className="mt-0 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-semibold">Resource Configurations</h2>
                    <Button
                      size="sm"
                      disabled={!selectedProject}
                      onClick={() => {
                        setCreateDialogType('resource')
                        setCreateDialogOpen(true)
                      }}
                    >
                      <Plus className="h-4 w-4 mr-1" />
                      Create
                    </Button>
                  </div>
                  <ResourceConfigList
                    resourcesQuery={resourcesQuery}
                    onSelectResource={handleSelectResource}
                    onCreateResource={() => {
                      setCreateDialogType('resource')
                      setCreateDialogOpen(true)
                    }}
                    selectedProject={selectedProject}
                  />
                </TabsContent>
                <TabsContent value="sources" className="mt-0 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-semibold">Message Sources</h2>
                    <Button
                      size="sm"
                      disabled={!selectedProject}
                      onClick={() => {
                        setCreateDialogType('message_source')
                        setCreateDialogOpen(true)
                      }}
                    >
                      <Plus className="h-4 w-4 mr-1" />
                      Create
                    </Button>
                  </div>
                  <MessageSourceConfigList
                    messageSourcesQuery={messageSourcesQuery}
                    onSelectSource={handleSelectSource}
                    onCreateSource={() => {
                      setCreateDialogType('message_source')
                      setCreateDialogOpen(true)
                    }}
                    selectedProject={selectedProject}
                  />
                </TabsContent>
                <TabsContent value="aliases" className="mt-0 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-semibold">Alias Bindings</h2>
                  </div>
                  <AliasesTab aliasesQuery={aliasesQuery} selectedProject={selectedProject} />
                </TabsContent>
              </div>
            </Tabs>
          </div>
        }
        right={
          <div className="h-full overflow-y-auto p-4 sm:p-6 px-4 sm:px-8">
            {detailError && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{detailError}</AlertDescription>
              </Alert>
            )}
            {!details.agent && !details.provider && !details.resource && !details.messageSource && (
              <div className="text-center py-16">
                <Eye className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No item selected</h3>
                <p className="text-muted-foreground">Select a configuration from the list to view and edit details.</p>
              </div>
            )}
            {(details.agent || details.provider || details.resource || details.messageSource) && (
              <Stack spacing="lg">
                {details.agent && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Agent: {details.agent.name}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <pre className="text-sm overflow-auto">{JSON.stringify(details.agent, null, 2)}</pre>
                    </CardContent>
                  </Card>
                )}
                {details.provider && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Provider: {details.provider.name}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <pre className="text-sm overflow-auto">{JSON.stringify(details.provider, null, 2)}</pre>
                    </CardContent>
                  </Card>
                )}
                {details.resource && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Resource: {details.resource.name}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <pre className="text-sm overflow-auto">{JSON.stringify(details.resource, null, 2)}</pre>
                    </CardContent>
                  </Card>
                )}
                {details.messageSource && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Message Source: {details.messageSource.name}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <pre className="text-sm overflow-auto">{JSON.stringify(details.messageSource, null, 2)}</pre>
                    </CardContent>
                  </Card>
                )}

                <ConfigEditor
                  projectId={selectedProject}
                  target={editorTarget}
                  content={editorContent}
                  onContentChange={setEditorContent}
                  baseHash={editorBaseHash}
                  setBaseHash={setEditorBaseHash}
                  diffResult={diffResult}
                  setDiffResult={setDiffResult}
                  setSuccess={setEditorSuccess}
                  setError={setEditorError}
                  queryClient={queryClient}
                />
                {editorError && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{editorError}</AlertDescription>
                  </Alert>
                )}
                {editorSuccess && (
                  <Alert variant="success">
                    <CheckCircle2 className="h-4 w-4" />
                    <AlertDescription>{editorSuccess}</AlertDescription>
                  </Alert>
                )}
              </Stack>
            )}
          </div>
        }
      />

      {/* Create Configuration Dialog */}
      <CreateConfigDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
        selectedProject={selectedProject}
        initialType={createDialogType}
        onSuccess={async ({ type, name }) => {
          // Switch to appropriate tab and refetch queries, then select the newly created config
          if (type === 'provider') {
            setActiveTab('providers')
            // Invalidate and wait for the query to refetch
            await queryClient.invalidateQueries({ queryKey: ['configs', 'providers', selectedProject] })
            try {
              const configs = await fetchProviderConfigs(selectedProject)
              const newConfig = configs.configs.find((c) => c.name === name)
              if (newConfig) {
                handleSelectProvider(newConfig)
              }
            } catch {
              // Silently fail - query invalidation will update the list
            }
          } else if (type === 'resource') {
            setActiveTab('resources')
            // Invalidate and wait for the query to refetch
            await queryClient.invalidateQueries({ queryKey: ['configs', 'resources', selectedProject] })
            try {
              const configs = await fetchResourceConfigs(selectedProject)
              const newConfig = configs.configs.find((c) => c.name === name)
              if (newConfig) {
                handleSelectResource(newConfig)
              }
            } catch {
              // Silently fail - query invalidation will update the list
            }
          } else if (type === 'message_source') {
            setActiveTab('sources')
            // Invalidate and wait for the query to refetch
            await queryClient.invalidateQueries({ queryKey: ['configs', 'messageSources', selectedProject] })
            try {
              const response = await fetchMessageSources(selectedProject)
              const newConfig = response.sources.find((c: MessageSourceSummary) => c.name === name)
              if (newConfig) {
                handleSelectSource(newConfig)
              }
            } catch {
              // Silently fail - query invalidation will update the list
            }
          }
        }}
      />
    </div>
  )
}

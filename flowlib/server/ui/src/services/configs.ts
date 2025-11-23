import api from './api'

/**
 * Union type for configuration categories
 */
export type ConfigType = 'provider' | 'resource'

export type AgentConfigSummary = {
  name: string
  persona: string
  allowed_tool_categories: string[]
  knowledge_plugins: string[]
  model_name: string
  llm_name: string
  temperature: number
  max_iterations: number
  enable_learning: boolean
  verbose: boolean
}

export type AgentConfigListResponse = {
  project_id: string
  configs: AgentConfigSummary[]
  total: number
}

export type ProviderConfigSummary = {
  name: string
  resource_type: string
  provider_type: string
  settings: Record<string, unknown>
}

export type ProviderConfigListResponse = {
  project_id: string
  configs: ProviderConfigSummary[]
  total: number
}

export type ResourceConfigSummary = {
  name: string
  resource_type: string
  metadata: Record<string, unknown>
}

export type ResourceConfigListResponse = {
  project_id: string
  configs: ResourceConfigSummary[]
  total: number
}

export type AliasEntry = {
  alias: string
  canonical: string
}

export type AliasListResponse = {
  project_id: string
  aliases: AliasEntry[]
  total: number
}

export type AliasApplyRequest = {
  project_id: string
  aliases: AliasEntry[]
}

export type ConfigDiffRequest = {
  project_id: string
  relative_path: string
  proposed_content: string
}

export type ConfigDiffResponse = {
  project_id: string
  relative_path: string
  exists: boolean
  diff: string[]
}

export type ConfigApplyRequest = {
  project_id: string
  relative_path: string
  content: string
  sha256_before: string
}

export type ConfigApplyResponse = {
  project_id: string
  relative_path: string
  sha256_before: string
  sha256_after: string
}

export type ProviderConfigCreatePayload = {
  project_id: string
  name: string
  resource_type: string
  provider_type: string
  description?: string
  settings: Record<string, unknown>
}

export type ResourceConfigCreatePayload = {
  project_id: string
  name: string
  resource_type: string
  provider_type: string
  description?: string
  config: Record<string, unknown>
}

export type SchemaField = {
  name: string
  type: string
  required: boolean
  description?: string | null
  default?: unknown
  enum?: string[] | null
  string_min_length?: number | null
  string_max_length?: number | null
  numeric_min?: number | null
  numeric_max?: number | null
  pattern?: string | null
  children?: SchemaField[] | null
  items_type?: string | null
  items_allowed_types?: string[] | null
  allowed_types?: string[] | null
}

export type SchemaResponse = {
  title: string
  fields: SchemaField[]
}

export type ProviderConfigUpdatePayload = {
  project_id: string
  name: string
  resource_type: string
  provider_type: string
  description?: string
  settings: Record<string, unknown>
}

export type ResourceConfigUpdatePayload = {
  project_id: string
  name: string
  resource_type: string
  provider_type: string
  description?: string
  config: Record<string, unknown>
}
const withProjectParam = (projectId: string) => ({
  params: { project_id: projectId },
})

export async function fetchAgentConfigs(
  projectId: string,
): Promise<AgentConfigListResponse> {
  const { data } = await api.get<AgentConfigListResponse>(
    '/configs/agents',
    withProjectParam(projectId),
  )
  return data
}

export async function fetchAgentConfig(
  projectId: string,
  configId: string,
): Promise<AgentConfigSummary> {
  const { data } = await api.get<AgentConfigSummary>(`/configs/agents/${configId}`, {
    params: { project_id: projectId },
  })
  return data
}

export async function fetchProviderConfigs(
  projectId: string,
): Promise<ProviderConfigListResponse> {
  const { data } = await api.get<ProviderConfigListResponse>(
    '/configs/providers',
    withProjectParam(projectId),
  )
  return data
}

export async function fetchResourceConfigs(
  projectId: string,
): Promise<ResourceConfigListResponse> {
  const { data } = await api.get<ResourceConfigListResponse>(
    '/configs/resources',
    withProjectParam(projectId),
  )
  return data
}

export async function fetchAliases(projectId: string): Promise<AliasListResponse> {
  const { data } = await api.get<AliasListResponse>('/configs/aliases', withProjectParam(projectId))
  return data
}

export async function applyAliases(payload: AliasApplyRequest): Promise<ConfigApplyResponse> {
  const { data } = await api.post<ConfigApplyResponse>('/configs/aliases/apply', payload)
  return data
}

export async function diffConfig(payload: ConfigDiffRequest): Promise<ConfigDiffResponse> {
  const { data } = await api.post<ConfigDiffResponse>('/diff/configs', payload)
  return data
}

export async function applyConfig(payload: ConfigApplyRequest): Promise<ConfigApplyResponse> {
  const { data } = await api.post<ConfigApplyResponse>('/diff/configs/apply', payload)
  return data
}

export async function createProviderConfig(
  payload: ProviderConfigCreatePayload,
): Promise<ConfigApplyResponse> {
  const { data } = await api.post<ConfigApplyResponse>('/configs/providers/create', payload)
  return data
}

export async function createResourceConfig(
  payload: ResourceConfigCreatePayload,
): Promise<ConfigApplyResponse> {
  const { data } = await api.post<ConfigApplyResponse>('/configs/resources/create', payload)
  return data
}

export async function fetchProviderSchema(
  resourceType: string,
  projectId: string,
  providerType?: string,
): Promise<SchemaResponse> {
  const { data } = await api.get<SchemaResponse>('/configs/providers/schema', {
    params: { resource_type: resourceType, project_id: projectId, provider_type: providerType },
  })
  return data
}

export async function fetchResourceSchema(
  resourceType: string,
  projectId: string,
  providerType?: string,
): Promise<SchemaResponse> {
  const { data } = await api.get<SchemaResponse>('/configs/resources/schema', {
    params: { resource_type: resourceType, project_id: projectId, provider_type: providerType },
  })
  return data
}

export async function listProviderTypes(resourceType: string): Promise<string[]> {
  const { data } = await api.get<string[]>('/configs/providers/types', { params: { resource_type: resourceType } })
  return data
}

export async function applyProviderStructured(
  payload: ProviderConfigUpdatePayload,
): Promise<ConfigApplyResponse> {
  const { data } = await api.post<ConfigApplyResponse>('/configs/providers/apply', payload)
  return data
}

export async function applyResourceStructured(
  payload: ResourceConfigUpdatePayload,
): Promise<ConfigApplyResponse> {
  const { data } = await api.post<ConfigApplyResponse>('/configs/resources/apply', payload)
  return data
}

export type RenderResponse = { content: string }

export async function renderProviderContent(payload: {
  name: string
  resource_type: string
  provider_type: string
  description?: string
  settings: Record<string, unknown>
}): Promise<RenderResponse> {
  const { data } = await api.post<RenderResponse>('/configs/providers/render', payload)
  return data
}

export async function renderResourceContent(payload: {
  name: string
  resource_type: string
  provider_type: string
  description?: string
  config: Record<string, unknown>
}): Promise<RenderResponse> {
  const { data } = await api.post<RenderResponse>('/configs/resources/render', payload)
  return data
}

export type AgentCreatePayload = {
  project_id: string
  name: string
  persona: string
  allowed_tool_categories: string[]
  knowledge_plugins?: string[]
  description?: string
}

export async function createAgentConfig(payload: AgentCreatePayload) {
  const { data } = await api.post<ConfigApplyResponse>('/configs/agents/create', payload)
  return data
}

export async function fetchAgentSchema(projectId: string) {
  const { data } = await api.get<SchemaResponse>('/configs/agents/schema', { params: { project_id: projectId } })
  return data
}

export async function applyAgentStructured(payload: {
  project_id: string
  name: string
  persona: string
  allowed_tool_categories: string[]
  knowledge_plugins?: string[]
  description?: string
  model_name?: string
  llm_name?: string
  temperature?: number
  max_iterations?: number
  enable_learning?: boolean
  verbose?: boolean
}) {
  const { data } = await api.post<ConfigApplyResponse>('/configs/agents/apply', payload)
  return data
}

export async function renderAgentContent(payload: {
  name: string
  persona: string
  allowed_tool_categories: string[]
  description?: string
}) {
  const { data } = await api.post<RenderResponse>('/configs/agents/render', payload)
  return data
}

export async function renameConfig(payload: {
  project_id: string
  old_relative_path: string
  new_relative_path: string
}): Promise<void> {
  await api.post('/configs/configs/rename', payload)
}

export async function deleteConfig(payload: {
  project_id: string
  relative_path: string
}): Promise<void> {
  await api.post('/configs/configs/delete', payload)
}


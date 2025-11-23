import api from './api'

export type ProjectMetadata = {
  id: string
  name: string
  description: string
  path: string
  created_at: string
  updated_at: string
}

export type ProjectListResponse = {
  projects: ProjectMetadata[]
  total: number
}

export type SetupType = 'empty' | 'guided' | 'fast'

export type FastSetupConfig = {
  llm_model_path: string
  embedding_model_path: string
  vector_db_path?: string | null
}

export type GuidedProviderConfig = {
  name: string
  resource_type: string
  provider_type: string
  description?: string
  settings?: Record<string, unknown>
}

export type GuidedResourceConfig = {
  name: string
  resource_type: string
  provider_type: string
  description?: string
  config?: Record<string, unknown>
}

export type GuidedSetupConfig = {
  providers?: GuidedProviderConfig[]
  resources?: GuidedResourceConfig[]
  aliases?: Record<string, string>
}

export type ProjectCreateRequest = {
  name: string
  description?: string
  setup_type?: SetupType
  fast_config?: FastSetupConfig | null
  guided_config?: GuidedSetupConfig | null
  agent_names?: string[]
  tool_categories?: string[]
}

export type ProjectValidationIssue = {
  path: string
  message: string
}

export type ProjectValidationResponse = {
  is_valid: boolean
  issues: ProjectValidationIssue[]
}

export async function fetchProjects(): Promise<ProjectListResponse> {
  const { data } = await api.get<ProjectListResponse>('/projects')
  return data
}

export async function createProject(payload: ProjectCreateRequest): Promise<ProjectMetadata> {
  const { data } = await api.post<ProjectMetadata>('/projects', payload)
  return data
}

export async function validateProject(projectId: string): Promise<ProjectValidationResponse> {
  const { data } = await api.post<ProjectValidationResponse>(`/projects/${projectId}/validate`)
  return data
}

export async function deleteProject(projectId: string): Promise<void> {
  await api.delete(`/projects/${projectId}`)
}


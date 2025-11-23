import api from './api'

export type AgentRunRequest = {
  project_id: string
  agent_config_name: string
  mode: string
  execution_config: Record<string, unknown>
}

export type AgentRunResponse = {
  run_id: string
  status: string
}

export type AgentRunStatusResponse = {
  run_id: string
  status: string
  started_at: string | null
  finished_at: string | null
  message: string | null
}

export type ReplSessionResponse = {
  session_id: string
  project_id: string
  agent_config_name: string
  created_at: string
}

export type ReplInputRequest = {
  message: string
}

export async function listAgents(projectId: string): Promise<string[]> {
  const { data } = await api.get<string[]>('/agents', { params: { project_id: projectId } })
  return data
}

export async function startAgentRun(payload: AgentRunRequest): Promise<AgentRunResponse> {
  const { data } = await api.post<AgentRunResponse>('/agents/run', payload)
  return data
}

export async function getRunStatus(runId: string): Promise<AgentRunStatusResponse> {
  const { data } = await api.get<AgentRunStatusResponse>(`/agents/runs/${runId}`)
  return data
}

export async function stopRun(runId: string): Promise<AgentRunStatusResponse> {
  const { data } = await api.post<AgentRunStatusResponse>(`/agents/runs/${runId}/stop`)
  return data
}

export async function listRuns(): Promise<AgentRunStatusResponse[]> {
  const { data } = await api.get<AgentRunStatusResponse[]>('/agents/runs')
  return data
}

export async function listRunHistory(limit = 100): Promise<AgentRunStatusResponse[]> {
  const { data } = await api.get<AgentRunStatusResponse[]>('/agents/runs/history', { params: { limit } })
  return data
}

export async function createReplSession(
  projectId: string,
  agentName: string,
): Promise<ReplSessionResponse> {
  const { data } = await api.post<ReplSessionResponse>('/agents/repl/sessions', {
    project_id: projectId,
    agent_config_name: agentName,
  })
  return data
}

export async function closeReplSession(sessionId: string): Promise<void> {
  await api.delete(`/agents/repl/sessions/${sessionId}`)
}

export async function sendReplInput(sessionId: string, message: string): Promise<void> {
  await api.post(`/agents/repl/sessions/${sessionId}/input`, { message })
}


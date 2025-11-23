import api from './api'

export type WorkspacePathResponse = {
  path: string
  message?: string | null
}

export type WorkspacePathRequest = {
  path: string
}

export type DirectoryEntry = {
  name: string
  path: string
  is_directory: boolean
  readable: boolean
}

export type DirectoryListingResponse = {
  path: string
  parent: string | null
  entries: DirectoryEntry[]
}

export async function getWorkspacePath(): Promise<WorkspacePathResponse> {
  const { data } = await api.get<WorkspacePathResponse>('/workspace/path')
  return data
}

export async function setWorkspacePath(path: string): Promise<WorkspacePathResponse> {
  const { data } = await api.put<WorkspacePathResponse>('/workspace/path', { path })
  return data
}

export async function browseDirectory(path: string | null = null): Promise<DirectoryListingResponse> {
  const params = path ? { path } : {}
  const { data } = await api.get<DirectoryListingResponse>('/workspace/browse', { params })
  return data
}

export type CreateDirectoryRequest = {
  parent_path: string
  name: string
}

export type CreateDirectoryResponse = {
  path: string
  message: string
}

export async function createDirectory(parentPath: string, name: string): Promise<CreateDirectoryResponse> {
  const { data } = await api.post<CreateDirectoryResponse>('/workspace/create-directory', {
    parent_path: parentPath,
    name,
  })
  return data
}


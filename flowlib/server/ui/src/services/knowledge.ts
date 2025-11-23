/**
 * Knowledge Plugin API Service
 *
 * Type-safe client for knowledge plugin operations.
 * Types match backend Pydantic models exactly - single source of truth.
 */

import api from './api'

// ========== ENUMS ==========

export enum DomainStrategy {
  GENERIC = 'generic',
  SOFTWARE_ENGINEERING = 'software_engineering',
  SCIENTIFIC_RESEARCH = 'scientific_research',
  BUSINESS_PROCESS = 'business_process',
  LEGAL_COMPLIANCE = 'legal_compliance',
}

export enum DocumentType {
  PDF = 'pdf',
  TXT = 'txt',
  EPUB = 'epub',
  MOBI = 'mobi',
  DOCX = 'docx',
  HTML = 'html',
  MARKDOWN = 'md',
}

// ========== STATISTICS TYPES ==========

export type ExtractionStats = {
  total_documents: number
  successful_documents: number
  failed_documents: number
  total_entities: number
  total_relationships: number
  total_chunks: number
  processing_time_seconds: number
}

export type PluginCapabilities = {
  has_vector_db: boolean
  has_graph_db: boolean
  supports_semantic_search: boolean
  supports_relationship_queries: boolean
}

// ========== PLUGIN SUMMARY AND DETAILS ==========

export type PluginSummary = {
  plugin_id: string
  name: string
  version: string
  description: string
  author: string
  domains: string[]
  created_at: string
  extraction_stats: ExtractionStats
  capabilities: PluginCapabilities
}

export type PluginListResponse = {
  project_id: string
  plugins: PluginSummary[]
  total: number
}

export type PluginDetails = PluginSummary & {
  plugin_path: string
  chunk_size: number
  chunk_overlap: number
  domain_strategy: DomainStrategy
  files_created: string[]
}

// ========== ENTITY AND RELATIONSHIP TYPES ==========

export type EntitySummary = {
  entity_id: string
  name: string
  entity_type: string
  description: string | null
  confidence: number
  frequency: number
  documents: string[]
}

export type RelationshipSummary = {
  relationship_id: string
  source_entity_id: string
  target_entity_id: string
  relationship_type: string
  description: string | null
  confidence: number
  frequency: number
  documents: string[]
}

export type DocumentSummary = {
  document_id: string
  file_name: string
  file_type: DocumentType
  word_count: number
  chunk_count: number
  summary: string | null
}

export type EntityListResponse = {
  plugin_id: string
  entities: EntitySummary[]
  total: number
}

export type RelationshipListResponse = {
  plugin_id: string
  relationships: RelationshipSummary[]
  total: number
}

export type DocumentListResponse = {
  plugin_id: string
  documents: DocumentSummary[]
  total: number
}

// ========== PLUGIN GENERATION TYPES ==========

export type PluginGenerationRequest = {
  project_id: string
  plugin_name: string
  input_directory: string
  domains: string[]
  description?: string
  author?: string
  version?: string
  use_vector_db?: boolean
  use_graph_db?: boolean
  max_files?: number
  chunk_size?: number
  chunk_overlap?: number
  domain_strategy?: DomainStrategy
}

export type PluginGenerationResponse = {
  success: boolean
  plugin_id: string
  plugin_path: string
  extraction_stats: ExtractionStats
  files_created: string[]
  message: string
  error_message: string | null
}

// ========== PLUGIN QUERY TYPES ==========

export type PluginQueryRequest = {
  query: string
  limit?: number
  entity_types?: string[]
  relationship_types?: string[]
  min_confidence?: number
}

export type QueryResultItem = {
  item_type: string
  item_id: string
  name: string
  description: string | null
  relevance_score: number
  metadata: Record<string, unknown>
}

export type PluginQueryResponse = {
  query: string
  results: QueryResultItem[]
  total_found: number
  processing_time_seconds: number
}

// ========== PLUGIN DELETION TYPES ==========

export type PluginDeleteResponse = {
  success: boolean
  plugin_id: string
  message: string
}

// ========== DOCUMENT UPLOAD TYPES ==========

export type DocumentUploadResult = {
  filename: string
  file_path: string
  file_size: number
  file_type: DocumentType
}

export type DocumentUploadResponse = {
  upload_id: string
  upload_directory: string
  uploaded_files: DocumentUploadResult[]
  total_uploaded: number
  total_size_bytes: number
}

// ========== API FUNCTIONS ==========

/**
 * List all knowledge plugins for a project
 */
export async function listPlugins(projectId: string): Promise<PluginListResponse> {
  const { data } = await api.get<PluginListResponse>('/knowledge/plugins', {
    params: { project_id: projectId },
  })
  return data
}

/**
 * Get detailed information about a specific plugin
 */
export async function getPluginDetails(
  projectId: string,
  pluginId: string,
): Promise<PluginDetails> {
  const { data } = await api.get<PluginDetails>(`/knowledge/plugins/${pluginId}`, {
    params: { project_id: projectId },
  })
  return data
}

/**
 * Generate a new knowledge plugin from documents
 */
export async function generatePlugin(
  request: PluginGenerationRequest,
): Promise<PluginGenerationResponse> {
  const { data } = await api.post<PluginGenerationResponse>('/knowledge/plugins/generate', request)
  return data
}

/**
 * Delete a knowledge plugin
 */
export async function deletePlugin(
  projectId: string,
  pluginId: string,
): Promise<PluginDeleteResponse> {
  const { data } = await api.delete<PluginDeleteResponse>(`/knowledge/plugins/${pluginId}`, {
    params: { project_id: projectId },
  })
  return data
}

/**
 * Query a knowledge plugin for relevant information
 */
export async function queryPlugin(
  projectId: string,
  pluginId: string,
  request: PluginQueryRequest,
): Promise<PluginQueryResponse> {
  const { data} = await api.post<PluginQueryResponse>(
    `/knowledge/plugins/${pluginId}/query`,
    request,
    { params: { project_id: projectId } },
  )
  return data
}

/**
 * List entities from a knowledge plugin
 */
export async function listEntities(
  projectId: string,
  pluginId: string,
  limit = 100,
  offset = 0,
): Promise<EntityListResponse> {
  const { data } = await api.get<EntityListResponse>(`/knowledge/plugins/${pluginId}/entities`, {
    params: { project_id: projectId, limit, offset },
  })
  return data
}

/**
 * List relationships from a knowledge plugin
 */
export async function listRelationships(
  projectId: string,
  pluginId: string,
  limit = 100,
  offset = 0,
): Promise<RelationshipListResponse> {
  const { data } = await api.get<RelationshipListResponse>(
    `/knowledge/plugins/${pluginId}/relationships`,
    { params: { project_id: projectId, limit, offset } },
  )
  return data
}

/**
 * List documents from a knowledge plugin
 */
export async function listDocuments(
  projectId: string,
  pluginId: string,
  limit = 100,
  offset = 0,
): Promise<DocumentListResponse> {
  const { data } = await api.get<DocumentListResponse>(`/knowledge/plugins/${pluginId}/documents`, {
    params: { project_id: projectId, limit, offset },
  })
  return data
}

/**
 * Upload documents for plugin generation
 */
export async function uploadDocuments(
  projectId: string,
  files: File[],
): Promise<DocumentUploadResponse> {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))

  const { data } = await api.post<DocumentUploadResponse>('/knowledge/documents/upload', formData, {
    params: { project_id: projectId },
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

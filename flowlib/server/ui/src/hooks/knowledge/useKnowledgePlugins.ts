/**
 * React hooks for knowledge plugin operations
 *
 * Provides type-safe, reactive data fetching and mutations for knowledge plugins
 * using React Query for caching and state management.
 */

import { useQuery, useMutation, useQueryClient, UseQueryResult, UseMutationResult } from '@tanstack/react-query'
import * as knowledgeApi from '../../services/knowledge'

// ========== QUERY HOOKS ==========

/**
 * Hook to fetch all knowledge plugins for a project
 */
export function usePluginList(projectId: string | null): UseQueryResult<knowledgeApi.PluginListResponse> {
  return useQuery({
    queryKey: ['knowledge', 'plugins', projectId],
    queryFn: () => {
      if (!projectId) throw new Error('Project ID is required')
      return knowledgeApi.listPlugins(projectId)
    },
    enabled: Boolean(projectId),
  })
}

/**
 * Hook to fetch detailed information about a specific plugin
 */
export function usePluginDetails(
  projectId: string | null,
  pluginId: string | null,
): UseQueryResult<knowledgeApi.PluginDetails> {
  return useQuery({
    queryKey: ['knowledge', 'plugin-details', projectId, pluginId],
    queryFn: () => {
      if (!projectId || !pluginId) throw new Error('Project ID and Plugin ID are required')
      return knowledgeApi.getPluginDetails(projectId, pluginId)
    },
    enabled: Boolean(projectId && pluginId),
  })
}

/**
 * Hook to fetch entities from a plugin
 */
export function usePluginEntities(
  projectId: string | null,
  pluginId: string | null,
  limit = 100,
  offset = 0,
): UseQueryResult<knowledgeApi.EntityListResponse> {
  return useQuery({
    queryKey: ['knowledge', 'entities', projectId, pluginId, limit, offset],
    queryFn: () => {
      if (!projectId || !pluginId) throw new Error('Project ID and Plugin ID are required')
      return knowledgeApi.listEntities(projectId, pluginId, limit, offset)
    },
    enabled: Boolean(projectId && pluginId),
  })
}

/**
 * Hook to fetch relationships from a plugin
 */
export function usePluginRelationships(
  projectId: string | null,
  pluginId: string | null,
  limit = 100,
  offset = 0,
): UseQueryResult<knowledgeApi.RelationshipListResponse> {
  return useQuery({
    queryKey: ['knowledge', 'relationships', projectId, pluginId, limit, offset],
    queryFn: () => {
      if (!projectId || !pluginId) throw new Error('Project ID and Plugin ID are required')
      return knowledgeApi.listRelationships(projectId, pluginId, limit, offset)
    },
    enabled: Boolean(projectId && pluginId),
  })
}

/**
 * Hook to fetch documents from a plugin
 */
export function usePluginDocuments(
  projectId: string | null,
  pluginId: string | null,
  limit = 100,
  offset = 0,
): UseQueryResult<knowledgeApi.DocumentListResponse> {
  return useQuery({
    queryKey: ['knowledge', 'documents', projectId, pluginId, limit, offset],
    queryFn: () => {
      if (!projectId || !pluginId) throw new Error('Project ID and Plugin ID are required')
      return knowledgeApi.listDocuments(projectId, pluginId, limit, offset)
    },
    enabled: Boolean(projectId && pluginId),
  })
}

// ========== MUTATION HOOKS ==========

/**
 * Hook to generate a new knowledge plugin
 */
export function useGeneratePlugin(): UseMutationResult<
  knowledgeApi.PluginGenerationResponse,
  Error,
  knowledgeApi.PluginGenerationRequest
> {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: knowledgeApi.PluginGenerationRequest) => knowledgeApi.generatePlugin(request),
    onSuccess: (_, variables) => {
      // Invalidate plugin list for the project
      queryClient.invalidateQueries({ queryKey: ['knowledge', 'plugins', variables.project_id] })
    },
  })
}

/**
 * Hook to delete a knowledge plugin
 */
export function useDeletePlugin(): UseMutationResult<
  knowledgeApi.PluginDeleteResponse,
  Error,
  { projectId: string; pluginId: string }
> {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ projectId, pluginId }) => knowledgeApi.deletePlugin(projectId, pluginId),
    onSuccess: (_, variables) => {
      // Invalidate plugin list and details
      queryClient.invalidateQueries({ queryKey: ['knowledge', 'plugins', variables.projectId] })
      queryClient.invalidateQueries({ queryKey: ['knowledge', 'plugin-details', variables.projectId, variables.pluginId] })
    },
  })
}

/**
 * Hook to query a knowledge plugin
 */
export function useQueryPlugin(): UseMutationResult<
  knowledgeApi.PluginQueryResponse,
  Error,
  { projectId: string; pluginId: string; request: knowledgeApi.PluginQueryRequest }
> {
  return useMutation({
    mutationFn: ({ projectId, pluginId, request }) => knowledgeApi.queryPlugin(projectId, pluginId, request),
  })
}

/**
 * Hook to upload documents for plugin generation
 */
export function useUploadDocuments(): UseMutationResult<
  knowledgeApi.DocumentUploadResponse,
  Error,
  { projectId: string; files: File[] }
> {
  return useMutation({
    mutationFn: ({ projectId, files }) => knowledgeApi.uploadDocuments(projectId, files),
  })
}

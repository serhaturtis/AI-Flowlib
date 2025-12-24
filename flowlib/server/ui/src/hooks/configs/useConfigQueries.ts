import { useQuery } from '@tanstack/react-query'
import {
  fetchAgentConfigs,
  fetchProviderConfigs,
  fetchResourceConfigs,
  fetchAliases,
  fetchMessageSources,
} from '../../services/configs'

/**
 * Custom hook to consolidate all config-related queries.
 * Provides centralized query management for configs page.
 *
 * @param selectedProject - Current selected project ID (queries are disabled if null)
 * @returns Object containing all query results
 */
export function useConfigQueries(selectedProject: string) {
  const agentsQuery = useQuery({
    queryKey: ['configs', 'agents', selectedProject],
    queryFn: () => fetchAgentConfigs(selectedProject),
    enabled: Boolean(selectedProject),
  })

  const providersQuery = useQuery({
    queryKey: ['configs', 'providers', selectedProject],
    queryFn: () => fetchProviderConfigs(selectedProject),
    enabled: Boolean(selectedProject),
  })

  const resourcesQuery = useQuery({
    queryKey: ['configs', 'resources', selectedProject],
    queryFn: () => fetchResourceConfigs(selectedProject),
    enabled: Boolean(selectedProject),
  })

  const aliasesQuery = useQuery({
    queryKey: ['configs', 'aliases', selectedProject],
    queryFn: () => fetchAliases(selectedProject),
    enabled: Boolean(selectedProject),
  })

  const messageSourcesQuery = useQuery({
    queryKey: ['configs', 'message-sources', selectedProject],
    queryFn: () => fetchMessageSources(selectedProject),
    enabled: Boolean(selectedProject),
  })

  return {
    agentsQuery,
    providersQuery,
    resourcesQuery,
    aliasesQuery,
    messageSourcesQuery,
  }
}

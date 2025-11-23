/**
 * Plugin Details Dialog Component
 *
 * Comprehensive view of plugin details with tabs for:
 * - Overview (metadata and stats)
 * - Entities (searchable table)
 * - Relationships (list view)
 * - Documents (list view)
 * - Query (search interface)
 */

import { useState } from 'react'
import { Brain, Database, Network, FileText, Search, X } from 'lucide-react'
import {
  usePluginDetails,
  usePluginEntities,
  usePluginRelationships,
  usePluginDocuments,
  useQueryPlugin,
} from '../../hooks/knowledge/useKnowledgePlugins'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui/Dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/Tabs'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Skeleton } from '../ui/Skeleton'
import { Alert, AlertDescription } from '../ui/Alert'
import { Badge } from '../ui/Badge'
import { Separator } from '../ui/Separator'

type Props = {
  projectId: string
  pluginId: string
  open: boolean
  onClose: () => void
}

export function PluginDetailsDialog({ projectId, pluginId, open, onClose }: Props) {
  const [searchQuery, setSearchQuery] = useState('')
  const [queryInput, setQueryInput] = useState('')

  const detailsQuery = usePluginDetails(projectId, pluginId)
  const entitiesQuery = usePluginEntities(projectId, pluginId)
  const relationshipsQuery = usePluginRelationships(projectId, pluginId)
  const documentsQuery = usePluginDocuments(projectId, pluginId)
  const queryMutation = useQueryPlugin()

  const handleQuery = () => {
    if (queryInput.trim()) {
      queryMutation.mutate({
        projectId,
        pluginId,
        request: { query: queryInput.trim() },
      })
    }
  }

  if (!open) return null

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <DialogTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              {detailsQuery.data?.name || 'Plugin Details'}
            </DialogTitle>
            <Button variant="ghost" size="sm" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        <Tabs defaultValue="overview" className="flex-1 overflow-hidden flex flex-col">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="entities">Entities</TabsTrigger>
            <TabsTrigger value="relationships">Relationships</TabsTrigger>
            <TabsTrigger value="documents">Documents</TabsTrigger>
            <TabsTrigger value="query">Query</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="flex-1 overflow-y-auto">
            {detailsQuery.isLoading && <Skeleton className="h-96" />}
            {detailsQuery.isError && (
              <Alert variant="destructive">
                <AlertDescription>Failed to load plugin details</AlertDescription>
              </Alert>
            )}
            {detailsQuery.data && (
              <div className="space-y-6">
                {/* Metadata */}
                <div>
                  <h3 className="font-semibold mb-3">Plugin Information</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Plugin ID:</span>
                      <span className="ml-2 font-mono">{detailsQuery.data.plugin_id}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Version:</span>
                      <span className="ml-2">{detailsQuery.data.version}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Author:</span>
                      <span className="ml-2">{detailsQuery.data.author}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Created:</span>
                      <span className="ml-2">
                        {new Date(detailsQuery.data.created_at).toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Description */}
                {detailsQuery.data.description && (
                  <>
                    <div>
                      <h3 className="font-semibold mb-2">Description</h3>
                      <p className="text-sm text-muted-foreground">{detailsQuery.data.description}</p>
                    </div>
                    <Separator />
                  </>
                )}

                {/* Domains */}
                <div>
                  <h3 className="font-semibold mb-2">Knowledge Domains</h3>
                  <div className="flex flex-wrap gap-2">
                    {detailsQuery.data.domains.map((domain) => (
                      <Badge key={domain}>{domain}</Badge>
                    ))}
                  </div>
                </div>

                <Separator />

                {/* Capabilities */}
                <div>
                  <h3 className="font-semibold mb-2">Capabilities</h3>
                  <div className="flex gap-4">
                    <div className="flex items-center gap-2">
                      <Database className={`h-4 w-4 ${detailsQuery.data.capabilities.has_vector_db ? 'text-green-500' : 'text-muted-foreground'}`} />
                      <span className="text-sm">Vector Database</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Network className={`h-4 w-4 ${detailsQuery.data.capabilities.has_graph_db ? 'text-green-500' : 'text-muted-foreground'}`} />
                      <span className="text-sm">Graph Database</span>
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Statistics */}
                <div>
                  <h3 className="font-semibold mb-3">Extraction Statistics</h3>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="p-4 bg-muted rounded-lg">
                      <div className="text-3xl font-bold">{detailsQuery.data.extraction_stats.total_documents}</div>
                      <div className="text-sm text-muted-foreground mt-1">Documents</div>
                      <div className="text-xs text-muted-foreground mt-1">
                        ({detailsQuery.data.extraction_stats.successful_documents} successful)
                      </div>
                    </div>
                    <div className="p-4 bg-muted rounded-lg">
                      <div className="text-3xl font-bold">{detailsQuery.data.extraction_stats.total_entities}</div>
                      <div className="text-sm text-muted-foreground mt-1">Entities</div>
                    </div>
                    <div className="p-4 bg-muted rounded-lg">
                      <div className="text-3xl font-bold">{detailsQuery.data.extraction_stats.total_relationships}</div>
                      <div className="text-sm text-muted-foreground mt-1">Relationships</div>
                    </div>
                  </div>
                </div>

                {/* Configuration */}
                <div>
                  <h3 className="font-semibold mb-2">Configuration</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Chunk Size:</span>
                      <span className="ml-2">{detailsQuery.data.chunk_size}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Chunk Overlap:</span>
                      <span className="ml-2">{detailsQuery.data.chunk_overlap}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Domain Strategy:</span>
                      <span className="ml-2">{detailsQuery.data.domain_strategy}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </TabsContent>

          {/* Entities Tab */}
          <TabsContent value="entities" className="flex-1 overflow-y-auto">
            {entitiesQuery.isLoading && <Skeleton className="h-96" />}
            {entitiesQuery.isError && (
              <Alert variant="destructive">
                <AlertDescription>Failed to load entities</AlertDescription>
              </Alert>
            )}
            {entitiesQuery.data && (
              <div className="space-y-4">
                <Input
                  placeholder="Search entities..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <div className="space-y-2">
                  {entitiesQuery.data.entities
                    .filter((entity) =>
                      entity.name.toLowerCase().includes(searchQuery.toLowerCase()),
                    )
                    .map((entity) => (
                      <div key={entity.entity_id} className="p-3 border rounded-lg">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-semibold">{entity.name}</h4>
                            <p className="text-sm text-muted-foreground">{entity.entity_type}</p>
                            {entity.description && (
                              <p className="text-sm mt-1">{entity.description}</p>
                            )}
                          </div>
                          <div className="text-right text-sm">
                            <div className="text-muted-foreground">Confidence</div>
                            <div className="font-medium">{(entity.confidence * 100).toFixed(0)}%</div>
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </TabsContent>

          {/* Relationships Tab */}
          <TabsContent value="relationships" className="flex-1 overflow-y-auto">
            {relationshipsQuery.isLoading && <Skeleton className="h-96" />}
            {relationshipsQuery.isError && (
              <Alert variant="destructive">
                <AlertDescription>Failed to load relationships</AlertDescription>
              </Alert>
            )}
            {relationshipsQuery.data && (
              <div className="space-y-2">
                {relationshipsQuery.data.relationships.map((rel) => (
                  <div key={rel.relationship_id} className="p-3 border rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="font-medium">{rel.source_entity_id}</span>
                      <span className="text-muted-foreground">→</span>
                      <Badge variant="secondary">{rel.relationship_type}</Badge>
                      <span className="text-muted-foreground">→</span>
                      <span className="font-medium">{rel.target_entity_id}</span>
                    </div>
                    {rel.description && (
                      <p className="text-sm text-muted-foreground">{rel.description}</p>
                    )}
                    <div className="flex justify-between mt-2 text-xs text-muted-foreground">
                      <span>Confidence: {(rel.confidence * 100).toFixed(0)}%</span>
                      <span>Frequency: {rel.frequency}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Documents Tab */}
          <TabsContent value="documents" className="flex-1 overflow-y-auto">
            {documentsQuery.isLoading && <Skeleton className="h-96" />}
            {documentsQuery.isError && (
              <Alert variant="destructive">
                <AlertDescription>Failed to load documents</AlertDescription>
              </Alert>
            )}
            {documentsQuery.data && (
              <div className="space-y-2">
                {documentsQuery.data.documents.map((doc) => (
                  <div key={doc.document_id} className="p-3 border rounded-lg">
                    <div className="flex items-start gap-3">
                      <FileText className="h-5 w-5 text-muted-foreground flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <h4 className="font-semibold">{doc.file_name}</h4>
                        <p className="text-sm text-muted-foreground mt-1">
                          {doc.word_count} words • {doc.chunk_count} chunks
                        </p>
                        {doc.summary && (
                          <p className="text-sm mt-2">{doc.summary}</p>
                        )}
                      </div>
                      <Badge>{doc.file_type}</Badge>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Query Tab */}
          <TabsContent value="query" className="flex-1 overflow-y-auto">
            <div className="space-y-4">
              <div className="flex gap-2">
                <Input
                  placeholder="Enter your query..."
                  value={queryInput}
                  onChange={(e) => setQueryInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
                />
                <Button onClick={handleQuery} disabled={queryMutation.isPending}>
                  <Search className="h-4 w-4 mr-2" />
                  Search
                </Button>
              </div>

              {queryMutation.isPending && <Skeleton className="h-64" />}

              {queryMutation.data && (
                <div className="space-y-4">
                  <div className="text-sm text-muted-foreground">
                    Found {queryMutation.data.total_found} results in{' '}
                    {queryMutation.data.processing_time_seconds.toFixed(2)}s
                  </div>
                  <div className="space-y-2">
                    {queryMutation.data.results.map((result, index) => (
                      <div key={index} className="p-3 border rounded-lg">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant="outline" className="text-xs">
                                {result.item_type}
                              </Badge>
                              <h4 className="font-semibold">{result.name}</h4>
                            </div>
                            {result.description && (
                              <p className="text-sm text-muted-foreground">{result.description}</p>
                            )}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {(result.relevance_score * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}

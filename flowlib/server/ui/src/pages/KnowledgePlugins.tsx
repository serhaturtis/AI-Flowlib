/**
 * Knowledge Plugins Page
 *
 * Main page for managing knowledge plugins - listing, creating, and viewing details.
 * Follows established patterns from Projects page with grid/list views and dialogs.
 */

import { useState } from 'react'
import { Brain, Plus, Grid3x3, List, AlertCircle } from 'lucide-react'
import { usePluginList, useDeletePlugin } from '../hooks/knowledge/useKnowledgePlugins'
import { useProjectContext } from '../contexts/ProjectContext'
import { Button } from '../components/ui/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card'
import { Alert, AlertDescription, AlertTitle } from '../components/ui/Alert'
import { Stack } from '../components/layout/Stack'
import { Skeleton } from '../components/ui/Skeleton'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../components/ui/Dialog'
import { PluginGenerationWizard } from '../components/knowledge/PluginGenerationWizard'
import { PluginGridView } from '../components/knowledge/PluginGridView'
import { PluginListView } from '../components/knowledge/PluginListView'
import { PluginDetailsDialog } from '../components/knowledge/PluginDetailsDialog'

export default function KnowledgePlugins() {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [selectedPluginId, setSelectedPluginId] = useState<string | null>(null)

  const { selectedProjectId } = useProjectContext()

  const { data, isLoading, isError, error } = usePluginList(selectedProjectId)
  const deleteMutation = useDeletePlugin()

  const handleDelete = async (pluginId: string) => {
    if (!selectedProjectId) return

    if (confirm('Are you sure you want to delete this plugin? This action cannot be undone.')) {
      try {
        await deleteMutation.mutateAsync({ projectId: selectedProjectId, pluginId })
      } catch (err) {
        console.error('Failed to delete plugin:', err)
      }
    }
  }

  const handleViewDetails = (pluginId: string) => {
    setSelectedPluginId(pluginId)
  }

  return (
    <Stack spacing="xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold tracking-tight">Knowledge Plugins</h1>
          <p className="text-muted-foreground mt-2">
            Generate and manage knowledge plugins from document collections
          </p>
        </div>
        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button disabled={!selectedProjectId}>
              <Plus className="h-4 w-4 mr-2" />
              Create Plugin
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Create Knowledge Plugin</DialogTitle>
              <DialogDescription>
                Generate a knowledge plugin from your documents with entity extraction and semantic search
              </DialogDescription>
            </DialogHeader>
            {selectedProjectId && (
              <PluginGenerationWizard
                projectId={selectedProjectId}
                onComplete={() => setCreateDialogOpen(false)}
                onCancel={() => setCreateDialogOpen(false)}
              />
            )}
          </DialogContent>
        </Dialog>
      </div>

      {/* Project selection warning */}
      {!selectedProjectId && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>No Project Selected</AlertTitle>
          <AlertDescription>
            Please select a project from the top bar to view and manage knowledge plugins.
          </AlertDescription>
        </Alert>
      )}

      {/* Plugin list */}
      {selectedProjectId && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Knowledge Plugins
                </CardTitle>
                <CardDescription>
                  {isLoading
                    ? 'Loading plugins…'
                    : `${data?.total ?? 0} plugin${data?.total === 1 ? '' : 's'} available`}
                </CardDescription>
              </div>
              {!isLoading && !isError && data && data.total > 0 && (
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant={viewMode === 'grid' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setViewMode('grid')}
                    title="Grid view"
                  >
                    <Grid3x3 className="h-4 w-4" />
                  </Button>
                  <Button
                    type="button"
                    variant={viewMode === 'list' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setViewMode('list')}
                    title="List view"
                  >
                    <List className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {isLoading && (
              <div className="space-y-4">
                <Skeleton className="h-32" />
                <Skeleton className="h-32" />
                <Skeleton className="h-32" />
              </div>
            )}

            {isError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>
                  {error instanceof Error ? error.message : 'Failed to load plugins'}
                </AlertDescription>
              </Alert>
            )}

            {!isLoading && !isError && data && data.total === 0 && (
              <div className="text-center py-12">
                <Brain className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Plugins Yet</h3>
                <p className="text-muted-foreground mb-4">
                  Create your first knowledge plugin from document collections
                </p>
                <Button onClick={() => setCreateDialogOpen(true)}>
                  <Plus className="h-4 w-4 mr-2" />
                  Create Plugin
                </Button>
              </div>
            )}

            {!isLoading && !isError && data && data.total > 0 && (
              <>
                {viewMode === 'grid' ? (
                  <PluginGridView
                    plugins={data.plugins}
                    onDelete={handleDelete}
                    onViewDetails={handleViewDetails}
                    isDeleting={deleteMutation.isPending}
                  />
                ) : (
                  <PluginListView
                    plugins={data.plugins}
                    onDelete={handleDelete}
                    onViewDetails={handleViewDetails}
                    isDeleting={deleteMutation.isPending}
                  />
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* Plugin details dialog */}
      {selectedPluginId && selectedProjectId && (
        <PluginDetailsDialog
          projectId={selectedProjectId}
          pluginId={selectedPluginId}
          open={Boolean(selectedPluginId)}
          onClose={() => setSelectedPluginId(null)}
        />
      )}
    </Stack>
  )
}

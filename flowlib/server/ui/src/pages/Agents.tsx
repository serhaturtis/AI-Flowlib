import { useState, useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Bot, Plus } from 'lucide-react'
import { listAgents } from '../services/agents'
import { fetchAgentSchema } from '../services/configs'
import { useProjectContext } from '../contexts/ProjectContext'
import { Card, CardDescription, CardHeader, CardTitle } from '../components/ui/Card'
import { Stack } from '../components/layout/Stack'
import { Button } from '../components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../components/ui/Dialog'
import { useAgentForm } from '../hooks/agents/useAgentForm'
import { useAgentEditor } from '../hooks/agents/useAgentEditor'
import { CreateAgentForm } from '../components/agents/CreateAgentForm'
import { AgentList } from '../components/agents/AgentList'
import { AgentEditor } from '../components/agents/AgentEditor'

export default function Agents() {
  const [createDialogOpen, setCreateDialogOpen] = useState(false)

  const {
    selectedProjectId: selectedProject,
    selectedProject: selectedProjectMeta,
    projects,
  } = useProjectContext()

  const agentsQuery = useQuery({
    queryKey: ['agents', selectedProject],
    queryFn: () => listAgents(selectedProject),
    enabled: Boolean(selectedProject),
  })

  const agentSchemaQuery = useQuery({
    queryKey: ['agent-schema', selectedProject],
    queryFn: () => fetchAgentSchema(selectedProject),
    enabled: Boolean(selectedProject),
  })

  const formHook = useAgentForm(selectedProject)
  const editorHook = useAgentEditor(selectedProject)

  // Track previous pending state to detect successful submission
  const prevIsPending = useRef(formHook.isPending)

  useEffect(() => {
    // If submission just finished (was pending, now not) and no error, close dialog
    if (prevIsPending.current && !formHook.isPending && !formHook.formError && formHook.formSuccess) {
      setCreateDialogOpen(false)
    }
    prevIsPending.current = formHook.isPending
  }, [formHook.isPending, formHook.formError, formHook.formSuccess])

  if (!selectedProject && projects.length > 0) {
    return (
      <Stack spacing="xl">
        <div>
          <h1 className="text-4xl font-bold tracking-tight flex items-center gap-2">
            <Bot className="h-8 w-8" />
            Agents
          </h1>
          <p className="text-muted-foreground mt-2">Create and manage Flowlib agents</p>
        </div>
        <Card>
          <CardHeader>
            <CardTitle>No Project Selected</CardTitle>
            <CardDescription>Select a project from the top bar to manage its agents</CardDescription>
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
              <Bot className="h-5 w-5 sm:h-6 sm:w-6" />
              Agents
            </h1>
            {selectedProjectMeta && (
              <p className="text-xs sm:text-sm text-muted-foreground mt-1 truncate">
                <span className="font-medium">Project:</span> {selectedProjectMeta.name} •{' '}
                <code className="bg-muted px-1.5 py-0.5 rounded text-xs">{selectedProjectMeta.path}</code>
              </p>
            )}
          </div>
          <div className="flex items-center gap-4 flex-shrink-0">
            <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
              <DialogTrigger asChild>
                <Button disabled={!selectedProject}>
                  <Plus className="h-4 w-4 mr-2" />
                  Create Agent
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create Agent</DialogTitle>
                  <DialogDescription>Create a new Flowlib agent configuration</DialogDescription>
                </DialogHeader>
                <CreateAgentForm formHook={formHook} projectId={selectedProject} />
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 sm:p-6 px-4 sm:px-8">
        <Stack spacing="xl">
          <AgentList
            agents={agentsQuery.data}
            isLoading={agentsQuery.isLoading}
            isError={agentsQuery.isError}
            error={agentsQuery.error as Error | null}
          />

          {agentSchemaQuery.data && (
            <AgentEditor
              editorHook={editorHook}
              schema={agentSchemaQuery.data}
              agents={agentsQuery.data ?? []}
            />
          )}
        </Stack>
      </div>
    </div>
  )
}

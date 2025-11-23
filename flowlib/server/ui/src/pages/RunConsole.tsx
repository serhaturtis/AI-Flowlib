import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Terminal } from 'lucide-react'

import { useProjectContext } from '../contexts/ProjectContext'
import { listAgents } from '../services/agents'

import { Card, CardDescription, CardHeader, CardTitle } from '../components/ui/Card'
import { Stack } from '../components/layout/Stack'
import { SplitPane } from '../components/layout/SplitPane'

import { useRunsPolling } from '../hooks/agents/useRunsPolling'
import { useReplSession } from '../hooks/agents/useReplSession'
import { useAgentRunForm } from '../hooks/agents/useAgentRunForm'

import { RunStartForm } from '../components/agents/RunStartForm'
import { ActiveRunsList } from '../components/agents/ActiveRunsList'
import { RunHistory } from '../components/agents/RunHistory'
import { RunDetails } from '../components/agents/RunDetails'
import { ReplSession } from '../components/agents/ReplSession'

export default function RunConsole() {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)

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

  // Custom hooks for state management
  const form = useAgentRunForm(
    (run) => {
      polling.addRun(run)
      setSelectedRunId(run.run_id)
    },
    (runId, updates) => polling.updateRun(runId, updates),
  )

  const polling = useRunsPolling(selectedRunId, form.agentName, form.mode)
  const repl = useReplSession()

  const selectedRun = selectedRunId ? polling.runs[selectedRunId] : null

  if (!selectedProject && projects.length > 0) {
    return (
      <Stack spacing="xl">
        <div>
          <h1 className="text-4xl font-bold tracking-tight flex items-center gap-2">
            <Terminal className="h-8 w-8" />
            Agent Run Console
          </h1>
          <p className="text-muted-foreground mt-2">Start, monitor, and interact with agent runs</p>
        </div>
        <Card>
          <CardHeader>
            <CardTitle>No Project Selected</CardTitle>
            <CardDescription>Select a project from the top bar to run agents</CardDescription>
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
              <Terminal className="h-5 w-5 sm:h-6 sm:w-6" />
              Agent Run Console
            </h1>
            {selectedProjectMeta && (
              <p className="text-xs sm:text-sm text-muted-foreground mt-1 truncate">
                <span className="font-medium">Project:</span> {selectedProjectMeta.name} •{' '}
                <code className="bg-muted px-1.5 py-0.5 rounded text-xs">{selectedProjectMeta.path}</code>
              </p>
            )}
          </div>
        </div>
      </div>

      <SplitPane
        leftWidth="450px"
        rightMinWidth="600px"
        className="flex-1 min-h-0"
        left={
          <div className="h-full overflow-y-auto p-4 sm:p-6 border-r border-border bg-muted/20">
            <Stack spacing="lg">
              <RunStartForm form={form} agentsQuery={agentsQuery} selectedProject={selectedProject} />
              <ActiveRunsList
                runs={polling.runs}
                selectedRunId={selectedRunId}
                onSelectRun={setSelectedRunId}
                stopMutation={form.stopMutation}
              />
              <RunHistory history={polling.history} />
            </Stack>
          </div>
        }
        right={
          <div className="h-full overflow-y-auto p-4 sm:p-6">
            <Stack spacing="lg">
              <RunDetails run={selectedRun} />
              <ReplSession repl={repl} selectedProject={selectedProject} agentName={form.agentName} />
            </Stack>
          </div>
        }
      />
    </div>
  )
}

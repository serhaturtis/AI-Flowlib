import { Play, AlertCircle, Timer } from 'lucide-react'
import { UseQueryResult } from '@tanstack/react-query'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { FormField } from '../forms/FormField'
import { Select } from '../ui/Select'
import { Input } from '../ui/Input'
import { Button } from '../ui/Button'
import { Alert, AlertDescription } from '../ui/Alert'
import { Spinner } from '../ui/Spinner'
import { Skeleton } from '../ui/Skeleton'
import { Stack } from '../layout/Stack'
import type { UseAgentRunFormResult } from '../../hooks/agents/useAgentRunForm'
import type { MessageSourceSummary } from '../../services/configs'

export interface RunStartFormProps {
  form: UseAgentRunFormResult
  agentsQuery: UseQueryResult<string[], Error>
  messageSourcesQuery?: UseQueryResult<MessageSourceSummary[], Error>
  selectedProject: string
}

/**
 * Form for starting a new agent run.
 *
 * Features:
 * - Agent selection
 * - Execution mode selection
 * - Max cycles configuration (for autonomous mode)
 * - Message source selection (for daemon mode)
 * - Form validation
 * - Loading states
 */
export function RunStartForm({ form, agentsQuery, messageSourcesQuery, selectedProject }: RunStartFormProps) {
  const {
    agentName,
    mode,
    maxCycles,
    selectedMessageSources,
    formError,
    setAgentName,
    setMode,
    setMaxCycles,
    setSelectedMessageSources,
    startMutation,
    handleStart,
  } = form

  const handleSourceToggle = (sourceName: string) => {
    if (selectedMessageSources.includes(sourceName)) {
      setSelectedMessageSources(selectedMessageSources.filter((s) => s !== sourceName))
    } else {
      setSelectedMessageSources([...selectedMessageSources, sourceName])
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Start Run
        </CardTitle>
        <CardDescription>Start a new agent run</CardDescription>
      </CardHeader>
      <CardContent>
        <Stack spacing="lg">
          <FormField label="Agent" required>
            {agentsQuery.isLoading ? (
              <Skeleton className="h-10 w-full" />
            ) : agentsQuery.isError ? (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{agentsQuery.error.message}</AlertDescription>
              </Alert>
            ) : (
              <Select value={agentName} onChange={(event) => setAgentName(event.target.value)}>
                <option value="">Select agent</option>
                {agentsQuery.data?.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </Select>
            )}
          </FormField>

          <FormField label="Mode" required>
            <Select value={mode} onChange={(event) => setMode(event.target.value)}>
              <option value="repl">REPL (Interactive)</option>
              <option value="autonomous">Autonomous</option>
              <option value="daemon">Daemon</option>
              <option value="remote">Remote</option>
            </Select>
          </FormField>

          {mode === 'autonomous' && (
            <FormField label="Max Cycles" description="Maximum number of execution cycles">
              <Input
                type="number"
                value={maxCycles}
                min={1}
                onChange={(event) => setMaxCycles(event.target.value)}
              />
            </FormField>
          )}

          {mode === 'daemon' && (
            <FormField
              label="Message Sources"
              description="Optional: Select sources to override agent defaults. If none selected, the agent's configured sources will be used."
            >
              {messageSourcesQuery?.isLoading ? (
                <Skeleton className="h-24 w-full" />
              ) : messageSourcesQuery?.isError ? (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{messageSourcesQuery.error.message}</AlertDescription>
                </Alert>
              ) : messageSourcesQuery?.data && messageSourcesQuery.data.length > 0 ? (
                <div className="border rounded-md p-3 space-y-2 max-h-48 overflow-y-auto">
                  {messageSourcesQuery.data.map((source) => (
                    <label
                      key={source.name}
                      className="flex items-center gap-3 p-2 hover:bg-muted/50 rounded cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={selectedMessageSources.includes(source.name)}
                        onChange={() => handleSourceToggle(source.name)}
                        className="h-4 w-4 rounded border-input"
                      />
                      <div className="flex items-center gap-2 flex-1 min-w-0">
                        <Timer className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                        <span className="font-medium truncate">{source.name}</span>
                        <span className="text-xs text-muted-foreground">({source.source_type})</span>
                        {!source.enabled && (
                          <span className="text-xs text-yellow-600">(disabled)</span>
                        )}
                      </div>
                    </label>
                  ))}
                </div>
              ) : (
                <div className="border rounded-md p-4 text-center text-muted-foreground">
                  <Timer className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No message sources configured for this project.</p>
                  <p className="text-xs mt-1">
                    The agent's default sources will be used, or create message sources in the
                    configs section.
                  </p>
                </div>
              )}
            </FormField>
          )}

          {formError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{formError}</AlertDescription>
            </Alert>
          )}

          <Button
            type="button"
            onClick={() => handleStart(selectedProject)}
            disabled={startMutation.isPending || !selectedProject || !agentName}
            className="w-full"
          >
            {startMutation.isPending ? (
              <>
                <Spinner size="sm" className="mr-2" />
                Starting…
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Start Run
              </>
            )}
          </Button>
        </Stack>
      </CardContent>
    </Card>
  )
}

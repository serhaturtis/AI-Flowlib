import { AlertCircle, CheckCircle2, XCircle, Settings } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Alert, AlertDescription, AlertTitle } from '../ui/Alert'
import { Button } from '../ui/Button'
import { Stack } from '../layout/Stack'
import { applyAliases } from '../../services/configs'
import type { ProjectValidationResponse } from '../../services/projects'

export interface ValidationResultsProps {
  result: {
    projectId: string
    data: ProjectValidationResponse
  }
}

/**
 * Display validation results for a project.
 */
export function ValidationResults({ result }: ValidationResultsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertCircle className="h-5 w-5" />
          Validation Results
        </CardTitle>
        <CardDescription>Project: {result.projectId}</CardDescription>
      </CardHeader>
      <CardContent>
        <Stack spacing="md">
          <div className="flex items-center gap-2">
            <span className="font-medium">Status:</span>
            {result.data.is_valid ? (
              <Badge variant="success" className="flex items-center gap-1">
                <CheckCircle2 className="h-3 w-3" />
                Valid
              </Badge>
            ) : (
              <Badge variant="destructive" className="flex items-center gap-1">
                <XCircle className="h-3 w-3" />
                Issues detected
              </Badge>
            )}
          </div>

          {result.data.issues.length === 0 ? (
            <Alert variant="success">
              <CheckCircle2 className="h-4 w-4" />
              <AlertDescription>No validation issues found.</AlertDescription>
            </Alert>
          ) : (
            <Stack spacing="md">
              {result.data.issues.map((issue, idx) => (
                <Alert key={`${issue.path}-${issue.message}-${idx}`} variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>
                    <code className="text-sm">{issue.path}</code>
                  </AlertTitle>
                  <AlertDescription>
                    <p className="mb-3">{issue.message}</p>
                    <div className="flex flex-wrap gap-2">
                      <Button type="button" variant="outline" size="sm" asChild>
                        <Link to={`/configs?project_id=${result.projectId}&open=aliases`}>
                          <Settings className="h-4 w-4 mr-1" />
                          Open Configs
                        </Link>
                      </Button>
                      {issue.message.toLowerCase().includes('alias') &&
                        issue.message.toLowerCase().includes('missing') && (
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={async () => {
                              try {
                                await applyAliases({
                                  project_id: result.projectId,
                                  aliases: [],
                                })
                                alert('aliases.py created.')
                              } catch (e) {
                                alert(
                                  'Failed to create aliases.py: ' +
                                    (e instanceof Error ? e.message : 'Unknown error'),
                                )
                              }
                            }}
                          >
                            Create aliases.py
                          </Button>
                        )}
                      {(issue.message.toLowerCase().includes('provider config') ||
                        issue.message.toLowerCase().includes('resource config') ||
                        issue.message.toLowerCase().includes('unexpected config file')) && (
                        <Button type="button" variant="outline" size="sm" asChild>
                          <Link to={`/configs?project_id=${result.projectId}&open=provider-create`}>
                            Go to Create Config
                          </Link>
                        </Button>
                      )}
                    </div>
                  </AlertDescription>
                </Alert>
              ))}
            </Stack>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

/**
 * Plugin Generation Wizard Component
 *
 * Multi-step wizard for creating knowledge plugins:
 * 1. Upload documents
 * 2. Configure plugin settings
 * 3. Generate plugin with progress tracking
 * 4. View completion summary
 */

import { useState } from 'react'
import { usePluginGenerationWizard } from '../../hooks/knowledge/usePluginGenerationWizard'
import { DomainStrategy } from '../../services/knowledge'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Label } from '../ui/Label'
import { Textarea } from '../ui/Textarea'
import { Alert, AlertDescription } from '../ui/Alert'
import { Separator } from '../ui/Separator'
import { Upload, FileText, Settings, Loader2, CheckCircle2, X, ArrowRight, ArrowLeft } from 'lucide-react'

type Props = {
  projectId: string
  onComplete: () => void
  onCancel: () => void
}

export function PluginGenerationWizard({ projectId, onComplete, onCancel }: Props) {
  const wizard = usePluginGenerationWizard()
  const [isDragging, setIsDragging] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    wizard.handleFilesSelected(files)
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    // Only set isDragging to false if leaving the drop zone entirely
    if (e.currentTarget.contains(e.relatedTarget as Node)) {
      return
    }
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      wizard.handleFilesSelected(files)
    }
  }

  const handleDomainAdd = (domain: string) => {
    if (domain.trim() && !wizard.config.domains.includes(domain.trim())) {
      wizard.updateConfig({
        domains: [...wizard.config.domains, domain.trim()],
      })
    }
  }

  const handleDomainRemove = (index: number) => {
    wizard.updateConfig({
      domains: wizard.config.domains.filter((_, i) => i !== index),
    })
  }

  return (
    <div className="space-y-6">
      {/* Progress Steps */}
      <div className="flex items-center justify-between mb-8">
        {['Upload', 'Configure', 'Generate', 'Complete'].map((step, index) => {
          const stepKeys: Array<typeof wizard.currentStep> = ['upload', 'configure', 'generate', 'complete']
          const currentIndex = stepKeys.indexOf(wizard.currentStep)
          const isActive = index === currentIndex
          const isCompleted = index < currentIndex

          return (
            <div key={step} className="flex items-center flex-1">
              <div className="flex flex-col items-center flex-1">
                <div
                  className={`
                    w-10 h-10 rounded-full flex items-center justify-center border-2
                    ${isActive ? 'border-primary bg-primary text-primary-foreground' : ''}
                    ${isCompleted ? 'border-primary bg-primary text-primary-foreground' : ''}
                    ${!isActive && !isCompleted ? 'border-muted bg-muted text-muted-foreground' : ''}
                  `}
                >
                  {isCompleted ? <CheckCircle2 className="h-5 w-5" /> : index + 1}
                </div>
                <span className="text-sm mt-2 font-medium">{step}</span>
              </div>
              {index < 3 && (
                <div
                  className={`h-0.5 flex-1 mx-2 ${
                    isCompleted ? 'bg-primary' : 'bg-muted'
                  }`}
                />
              )}
            </div>
          )
        })}
      </div>

      {/* Error Display */}
      {wizard.error && (
        <Alert variant="destructive">
          <AlertDescription>{wizard.error}</AlertDescription>
        </Alert>
      )}

      {/* Step 1: Upload Documents */}
      {wizard.currentStep === 'upload' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2">Upload Documents</h3>
            <p className="text-sm text-muted-foreground">
              Select documents to process. Supported formats: PDF, TXT, EPUB, MOBI, DOCX, HTML, Markdown
            </p>
          </div>

          <div
            className={`
              border-2 border-dashed rounded-lg p-8 text-center transition-colors
              ${isDragging
                ? 'border-primary bg-primary/5'
                : 'border-muted hover:border-primary/50'
              }
            `}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Upload className={`h-12 w-12 mx-auto mb-4 ${isDragging ? 'text-primary' : 'text-muted-foreground'}`} />
            <Label htmlFor="file-upload" className="cursor-pointer">
              <span className="text-sm font-medium">
                {isDragging ? 'Drop files here' : 'Choose files or drag and drop'}
              </span>
              <Input
                id="file-upload"
                type="file"
                multiple
                accept=".pdf,.txt,.epub,.mobi,.docx,.html,.md"
                onChange={handleFileChange}
                className="hidden"
              />
            </Label>
          </div>

          {wizard.uploadedFiles.length > 0 && (
            <div className="space-y-2">
              <Label>Selected Files ({wizard.uploadedFiles.length})</Label>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {wizard.uploadedFiles.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-2 bg-muted rounded text-sm"
                  >
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <FileText className="h-4 w-4 flex-shrink-0" />
                      <span className="truncate">{file.name}</span>
                      <span className="text-muted-foreground flex-shrink-0">
                        ({(file.size / 1024).toFixed(1)} KB)
                      </span>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => wizard.removeFile(index)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex justify-between pt-4">
            <Button type="button" variant="outline" onClick={onCancel}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => wizard.uploadFiles(projectId)}
              disabled={wizard.uploadedFiles.length === 0 || wizard.isUploading}
            >
              {wizard.isUploading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  Next
                  <ArrowRight className="h-4 w-4 ml-2" />
                </>
              )}
            </Button>
          </div>
        </div>
      )}

      {/* Step 2: Configure Plugin */}
      {wizard.currentStep === 'configure' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2">Configure Plugin</h3>
            <p className="text-sm text-muted-foreground">
              Set up your knowledge plugin with custom settings
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <Label htmlFor="plugin-name">Plugin Name *</Label>
              <Input
                id="plugin-name"
                value={wizard.config.plugin_name}
                onChange={(e) =>
                  wizard.updateConfig({ plugin_name: e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, '') })
                }
                placeholder="my-knowledge-plugin"
                className="mt-1"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Lowercase, alphanumeric, hyphens, and underscores only
              </p>
            </div>

            <div>
              <Label>Knowledge Domains *</Label>
              <div className="flex gap-2 mt-1">
                <Input
                  placeholder="Add domain (e.g., technology, history)"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      handleDomainAdd(e.currentTarget.value)
                      e.currentTarget.value = ''
                    }
                  }}
                />
              </div>
              {wizard.config.domains.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                  {wizard.config.domains.map((domain, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-1 px-2 py-1 bg-primary/10 text-primary rounded text-sm"
                    >
                      {domain}
                      <button
                        type="button"
                        onClick={() => handleDomainRemove(index)}
                        className="hover:bg-primary/20 rounded p-0.5"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div>
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={wizard.config.description}
                onChange={(e) => wizard.updateConfig({ description: e.target.value })}
                placeholder="Describe what this plugin contains..."
                className="mt-1"
                rows={3}
              />
            </div>

            <div>
              <Label htmlFor="author">Author</Label>
              <Input
                id="author"
                value={wizard.config.author}
                onChange={(e) => wizard.updateConfig({ author: e.target.value })}
                className="mt-1"
              />
            </div>

            <Separator />

            <div>
              <Label htmlFor="domain-strategy">Domain Strategy</Label>
              <select
                id="domain-strategy"
                value={wizard.config.domain_strategy}
                onChange={(e) =>
                  wizard.updateConfig({ domain_strategy: e.target.value as DomainStrategy })
                }
                className="w-full mt-1 px-3 py-2 border border-input rounded-md"
              >
                <option value={DomainStrategy.GENERIC}>Generic</option>
                <option value={DomainStrategy.SOFTWARE_ENGINEERING}>Software Engineering</option>
                <option value={DomainStrategy.SCIENTIFIC_RESEARCH}>Scientific Research</option>
                <option value={DomainStrategy.BUSINESS_PROCESS}>Business Process</option>
                <option value={DomainStrategy.LEGAL_COMPLIANCE}>Legal Compliance</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="chunk-size">Chunk Size</Label>
                <Input
                  id="chunk-size"
                  type="number"
                  min={100}
                  max={5000}
                  value={wizard.config.chunk_size}
                  onChange={(e) => wizard.updateConfig({ chunk_size: Number(e.target.value) })}
                  className="mt-1"
                />
              </div>
              <div>
                <Label htmlFor="chunk-overlap">Chunk Overlap</Label>
                <Input
                  id="chunk-overlap"
                  type="number"
                  min={0}
                  max={1000}
                  value={wizard.config.chunk_overlap}
                  onChange={(e) => wizard.updateConfig({ chunk_overlap: Number(e.target.value) })}
                  className="mt-1"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={wizard.config.use_vector_db}
                  onChange={(e) => wizard.updateConfig({ use_vector_db: e.target.checked })}
                  className="rounded"
                />
                <span className="text-sm">Enable Vector Database (Semantic Search)</span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={wizard.config.use_graph_db}
                  onChange={(e) => wizard.updateConfig({ use_graph_db: e.target.checked })}
                  className="rounded"
                />
                <span className="text-sm">Enable Graph Database (Relationship Queries)</span>
              </label>
            </div>
          </div>

          <div className="flex justify-between pt-4">
            <Button type="button" variant="outline" onClick={wizard.prevStep}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button type="button" onClick={wizard.nextStep}>
              Next
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </div>
      )}

      {/* Step 3: Generate Plugin */}
      {wizard.currentStep === 'generate' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2">Generate Plugin</h3>
            <p className="text-sm text-muted-foreground">
              Review settings and start generation
            </p>
          </div>

          <div className="space-y-2 p-4 bg-muted rounded-lg">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Plugin Name:</span>
              <span className="font-medium">{wizard.config.plugin_name}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Domains:</span>
              <span className="font-medium">{wizard.config.domains.join(', ')}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Files:</span>
              <span className="font-medium">{wizard.uploadedFiles.length}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Vector DB:</span>
              <span className="font-medium">{wizard.config.use_vector_db ? 'Enabled' : 'Disabled'}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Graph DB:</span>
              <span className="font-medium">{wizard.config.use_graph_db ? 'Enabled' : 'Disabled'}</span>
            </div>
          </div>

          {wizard.isGenerating && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Generating plugin... This may take several minutes.</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all duration-300"
                  style={{ width: `${wizard.generationProgress}%` }}
                />
              </div>
            </div>
          )}

          <div className="flex justify-between pt-4">
            <Button type="button" variant="outline" onClick={wizard.prevStep} disabled={wizard.isGenerating}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button
              type="button"
              onClick={() => wizard.generatePlugin(projectId)}
              disabled={wizard.isGenerating}
            >
              {wizard.isGenerating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Settings className="h-4 w-4 mr-2" />
                  Generate Plugin
                </>
              )}
            </Button>
          </div>
        </div>
      )}

      {/* Step 4: Complete */}
      {wizard.currentStep === 'complete' && (
        <div className="space-y-4 text-center">
          <CheckCircle2 className="h-16 w-16 mx-auto text-green-500" />
          <div>
            <h3 className="text-lg font-semibold mb-2">Plugin Created Successfully!</h3>
            <p className="text-sm text-muted-foreground">
              Your knowledge plugin has been generated and is ready to use
            </p>
          </div>

          <div className="flex justify-center gap-2 pt-4">
            <Button type="button" variant="outline" onClick={wizard.reset}>
              Create Another
            </Button>
            <Button type="button" onClick={onComplete}>
              Done
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

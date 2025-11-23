import { ProjectCard } from './ProjectCard'
import type { ProjectMetadata } from '../../services/projects'

export interface ProjectGridViewProps {
  projects: ProjectMetadata[]
  isValidating: boolean
  validatingProjectId: string | undefined
  onValidate: (projectId: string) => void
  onDelete: (projectId: string) => void
  deletingProjectId?: string
}

/**
 * Grid view layout for projects.
 */
export function ProjectGridView({
  projects,
  isValidating,
  validatingProjectId,
  onValidate,
  onDelete,
  deletingProjectId,
}: ProjectGridViewProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {projects.map((project) => (
        <ProjectCard
          key={project.id}
          project={project}
          isValidating={isValidating && validatingProjectId === project.id}
          onValidate={onValidate}
          onDelete={onDelete}
          isDeleting={deletingProjectId === project.id}
        />
      ))}
    </div>
  )
}

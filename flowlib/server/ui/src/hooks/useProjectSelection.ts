import { useEffect, useState, useCallback } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'

const STORAGE_KEY = 'flowlib_active_project_id'

/**
 * Hook for managing project selection with persistence.
 * Priority: URL param > localStorage > first available project
 */
export function useProjectSelection(availableProjects: Array<{ id: string; name: string }>) {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [selectedProject, setSelectedProjectState] = useState<string>('')

  // Initialize from URL param, localStorage, or first available
  useEffect(() => {
    const urlProject = searchParams.get('project')
    if (urlProject && availableProjects.some((p) => p.id === urlProject)) {
      setSelectedProjectState(urlProject)
      try {
        localStorage.setItem(STORAGE_KEY, urlProject)
      } catch (error) {
        console.error('Failed to save project selection to localStorage:', error)
      }
      return
    }

    if (availableProjects.length > 0) {
      try {
        const stored = localStorage.getItem(STORAGE_KEY)
        if (stored && availableProjects.some((p) => p.id === stored)) {
          setSelectedProjectState(stored)
          // Update URL to reflect localStorage
          const newParams = new URLSearchParams(searchParams)
          newParams.set('project', stored)
          navigate({ search: newParams.toString() }, { replace: true })
          return
        }
      } catch (error) {
        console.error('Failed to read project selection from localStorage:', error)
      }

      // Fallback to first available project
      const firstProject = availableProjects[0].id
      setSelectedProjectState(firstProject)
      try {
        localStorage.setItem(STORAGE_KEY, firstProject)
      } catch (error) {
        console.error('Failed to save default project to localStorage:', error)
      }
      const newParams = new URLSearchParams(searchParams)
      newParams.set('project', firstProject)
      navigate({ search: newParams.toString() }, { replace: true })
    }
  }, [availableProjects, searchParams, navigate])

  // Update project selection
  const setSelectedProject = useCallback(
    (projectId: string) => {
      if (!availableProjects.some((p) => p.id === projectId)) {
        return
      }
      setSelectedProjectState(projectId)
      try {
        localStorage.setItem(STORAGE_KEY, projectId)
      } catch (error) {
        console.error('Failed to update project selection in localStorage:', error)
      }
      // Update URL param
      const newParams = new URLSearchParams(searchParams)
      newParams.set('project', projectId)
      navigate({ search: newParams.toString() }, { replace: true })
    },
    [availableProjects, searchParams, navigate],
  )

  return [selectedProject, setSelectedProject] as const
}


import { useEffect, useState, useCallback } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'

/**
 * Hook for syncing state with URL search params.
 * Updates URL when state changes and restores state from URL on mount.
 */
export function useUrlState<T extends string>(
  paramName: string,
  defaultValue: T,
  validValues?: T[],
): [T, (value: T) => void] {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [state, setState] = useState<T>(() => {
    const urlValue = searchParams.get(paramName)
    if (urlValue && (!validValues || validValues.includes(urlValue as T))) {
      return urlValue as T
    }
    return defaultValue
  })

  // Sync URL param on mount if not present
  useEffect(() => {
    const urlValue = searchParams.get(paramName)
    if (!urlValue || (validValues && !validValues.includes(urlValue as T))) {
      const newParams = new URLSearchParams(searchParams)
      newParams.set(paramName, defaultValue)
      navigate({ search: newParams.toString() }, { replace: true })
    }
  }, [paramName, defaultValue, validValues, searchParams, navigate])

  // Update state when URL param changes
  useEffect(() => {
    const urlValue = searchParams.get(paramName)
    if (urlValue && (!validValues || validValues.includes(urlValue as T))) {
      setState(urlValue as T)
    } else if (!urlValue && state !== defaultValue) {
      // URL param was removed, restore default
      setState(defaultValue)
    }
  }, [paramName, searchParams, validValues, state, defaultValue])

  // Update URL when state changes
  const setValue = useCallback(
    (value: T) => {
      setState(value)
      const newParams = new URLSearchParams(searchParams)
      if (value === defaultValue) {
        newParams.delete(paramName)
      } else {
        newParams.set(paramName, value)
      }
      navigate({ search: newParams.toString() }, { replace: true })
    },
    [paramName, defaultValue, searchParams, navigate],
  )

  return [state, setValue]
}


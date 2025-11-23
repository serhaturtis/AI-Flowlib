import axios from 'axios'

/**
 * HTTP client configuration constants
 */
const HTTP_TIMEOUTS = {
  /** Default timeout for most API requests (30 seconds) */
  DEFAULT: 30000,
  /** Extended timeout for long-running operations like agent runs (5 minutes) */
  LONG_RUNNING: 300000,
} as const

/**
 * Axios HTTP client with configured timeouts and error handling
 */
const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
  // Default timeout for all requests
  timeout: HTTP_TIMEOUTS.DEFAULT,
  // Validate status codes - reject on any error status
  validateStatus: (status) => status >= 200 && status < 300,
})

// Add response interceptor for consistent error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === 'ECONNABORTED' && error.message.includes('timeout')) {
      // Enhance timeout errors with helpful message
      error.message = `Request timeout after ${error.config?.timeout || HTTP_TIMEOUTS.DEFAULT}ms. The server may be slow or unresponsive.`
    }
    return Promise.reject(error)
  }
)

export default api
export { HTTP_TIMEOUTS }


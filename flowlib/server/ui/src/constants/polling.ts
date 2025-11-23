/**
 * Polling interval constants (in milliseconds).
 * Centralized to make testing and configuration easier.
 */

export const POLLING_INTERVALS = {
  /** Poll active runs status */
  ACTIVE_RUNS: 5000,
  /** Poll run history */
  RUN_HISTORY: 7000,
  /** Poll selected run status (more frequent) */
  SELECTED_RUN: 2000,
  /** WebSocket reconnect delay */
  WEBSOCKET_RECONNECT: 1500,
  /** Auto-close dialog after success */
  SUCCESS_AUTO_CLOSE: 1500,
} as const

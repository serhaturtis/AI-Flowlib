"""
Async-Qt integration helper for proper event loop management.

Provides utilities for running async operations in Qt GUI context
while following flowlib's async-first design principles.
"""

import asyncio
import logging
import threading
from typing import Callable, Any, Optional, Awaitable
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QObject, Signal, QTimer

logger = logging.getLogger(__name__)


class AsyncQtHelper(QObject):
    """Helper class for managing async operations in Qt GUI context."""
    
    # Signals for async operation completion
    async_operation_completed = Signal(str, bool, object)  # operation_id, success, result
    async_operation_failed = Signal(str, str)  # operation_id, error_message
    
    def __init__(self):
        super().__init__()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="async_qt")
        self._running_operations = {}
        
    def run_async_operation(self, operation_id: str, coroutine: Awaitable[Any]):
        """
        Run an async operation in a separate thread with proper event loop.
        
        This follows flowlib's async-first design while being compatible with Qt.
        """
        if operation_id in self._running_operations:
            logger.warning(f"Operation {operation_id} already running")
            return
        
        def run_in_thread():
            """Run the coroutine in a new event loop in a separate thread."""
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(coroutine)
                    self.async_operation_completed.emit(operation_id, True, result)
                    logger.debug(f"Async operation {operation_id} completed successfully")
                finally:
                    loop.close()
                    
            except Exception as e:
                logger.error(f"Async operation {operation_id} failed: {e}")
                self.async_operation_failed.emit(operation_id, str(e))
            finally:
                # Clean up tracking
                if operation_id in self._running_operations:
                    del self._running_operations[operation_id]
        
        # Submit to thread pool
        future = self._executor.submit(run_in_thread)
        self._running_operations[operation_id] = future
        
        logger.debug(f"Started async operation {operation_id}")
    
    def run_async_with_callback(self, coroutine: Awaitable[Any], 
                               success_callback: Optional[Callable] = None,
                               error_callback: Optional[Callable] = None):
        """
        Run async operation with callbacks.
        
        Convenience method for simple async operations.
        """
        operation_id = f"callback_op_{id(coroutine)}"
        
        def on_completed(op_id: str, success: bool, result: Any):
            if op_id == operation_id and success and success_callback:
                success_callback(result)
        
        def on_failed(op_id: str, error: str):
            if op_id == operation_id and error_callback:
                error_callback(error)
        
        # Connect temporary signals
        self.async_operation_completed.connect(on_completed)
        self.async_operation_failed.connect(on_failed)
        
        self.run_async_operation(operation_id, coroutine)
    
    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a running async operation."""
        if operation_id in self._running_operations:
            future = self._running_operations[operation_id]
            if future.cancel():
                del self._running_operations[operation_id]
                logger.info(f"Cancelled async operation {operation_id}")
                return True
        return False
    
    def is_operation_running(self, operation_id: str) -> bool:
        """Check if an async operation is currently running."""
        return operation_id in self._running_operations
    
    def get_running_operations(self) -> list:
        """Get list of currently running operation IDs."""
        return list(self._running_operations.keys())
    
    def shutdown(self):
        """Shutdown the async helper and cleanup resources."""
        logger.info("Shutting down AsyncQtHelper")
        
        # Cancel all running operations
        for operation_id in list(self._running_operations.keys()):
            self.cancel_operation(operation_id)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        logger.info("AsyncQtHelper shutdown complete")


class AsyncServiceMixin:
    """
    Mixin class for services that need async operations in Qt context.
    
    Provides common async utilities following flowlib patterns.
    """
    
    def __init__(self):
        # Ensure proper initialization following CLAUDE.md principles
        if not hasattr(self, '_async_helper_initialized') or not self._async_helper_initialized:
            self._async_helper = AsyncQtHelper()
            self._async_helper_initialized = True
    
    def run_async(self, operation_id: str, coroutine: Awaitable[Any]):
        """Run an async operation using the helper."""
        self._async_helper.run_async_operation(operation_id, coroutine)
    
    def run_async_with_callback(self, coroutine: Awaitable[Any],
                               success_callback: Optional[Callable] = None,
                               error_callback: Optional[Callable] = None):
        """Run async operation with callbacks."""
        self._async_helper.run_async_with_callback(coroutine, success_callback, error_callback)
    
    def cleanup_async(self):
        """Cleanup async operations following CLAUDE.md principles."""
        # Use proper attribute check with initialization flag
        if hasattr(self, '_async_helper_initialized') and self._async_helper_initialized and hasattr(self, '_async_helper'):
            self._async_helper.shutdown()
            self._async_helper_initialized = False


def async_qt_safe(func: Callable) -> Callable:
    """
    Decorator to make async functions Qt-safe.
    
    Ensures async functions run in proper event loop context.
    """
    def wrapper(*args, **kwargs):
        async def run_async():
            return await func(*args, **kwargs)
        
        # If we're in a Qt context, use helper
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            # If we're in a running loop (Qt main thread), use thread execution
            helper = AsyncQtHelper()
            operation_id = f"decorated_{func.__name__}_{id(args)}"
            helper.run_async_operation(operation_id, run_async())
        except RuntimeError:
            # No running loop, safe to create one
            return asyncio.run(run_async())
    
    return wrapper


def ensure_async_context(coroutine: Awaitable[Any]) -> Any:
    """
    Ensure async operation runs in proper context.
    
    Utility function for running async operations from sync Qt code.
    """
    try:
        loop = asyncio.get_running_loop()
        # In Qt context, use thread
        import threading
        result = [None]
        exception = [None]
        
        def run_in_thread():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result[0] = new_loop.run_until_complete(coroutine)
                finally:
                    new_loop.close()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if exception[0]:
            raise exception[0]
        return result[0]
        
    except RuntimeError:
        # No running loop, safe to create one
        return asyncio.run(coroutine)
"""
Base controller class for business logic controllers.

Clean implementation following CLAUDE.md principles:
- No fallbacks, no workarounds, single Pydantic contracts
- Type safety everywhere with strict validation
- Async-first design with proper error handling
- No legacy code, no backward compatibility
"""

import logging
import asyncio
from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, Union
from pydantic import Field
from flowlib.core.models import StrictBaseModel, MutableStrictBaseModel

from PySide6.QtCore import QObject, Signal, QThread

from .models import OperationResult

logger = logging.getLogger(__name__)


class ControllerState(MutableStrictBaseModel):
    """Base controller state with strict validation but mutable for runtime updates."""
    # Inherits strict configuration from MutableStrictBaseModel
    
    initialized: bool = False
    current_operations: int = 0
    last_operation: Optional[str] = None


class QObjectMeta(type(QObject), ABCMeta):
    """Metaclass that combines QObject and ABC metaclasses."""
    pass


class BaseController(QObject, metaclass=QObjectMeta):
    """
    Base class for all business logic controllers.
    
    Async-first design with strict contracts, no fallbacks.
    """
    
    # Common signals for view communication
    operation_started = Signal(str)  # operation_name
    operation_completed = Signal(str, bool, object)  # operation_name, success, result
    operation_failed = Signal(str, str)  # operation_name, error_message
    progress_updated = Signal(str, int)  # operation_name, percentage
    status_updated = Signal(str)  # status_message
    
    def __init__(self, service_factory=None, parent=None):
        super().__init__(parent)
        self.service_factory = service_factory
        self.state = ControllerState()
        self._current_workers = {}
        
        logger.debug(f"{self.__class__.__name__} controller created")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the controller and its services."""
        pass
    
    def is_service_available(self) -> bool:
        """Check if the controller's services are available for operations."""
        return self.state.initialized
    
    def on_service_created(self, service_name: str, service_instance: Union[str, int, float, bool, dict, list, object]) -> None:
        """Handle service creation events."""
        logger.debug(f"Service created: {service_name}")
    
    def on_service_failed(self, service_name: str, error_message: str) -> None:
        """Handle service creation failures."""
        logger.warning(f"Service failed: {service_name} - {error_message}")
        self.operation_failed.emit(service_name, error_message)
    
    def start_operation(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> None:
        """Start an async operation using worker thread."""
        if operation_name in self._current_workers:
            logger.warning(f"Operation {operation_name} already running")
            return
        
        self.operation_started.emit(operation_name)
        self.state.current_operations += 1
        self.state.last_operation = operation_name
        
        worker = OperationWorker(operation_func, *args, **kwargs)
        
        # Create explicit handler functions
        def on_finished(result):
            logger.info(f"Operation {operation_name} finished, result type: {type(result)}")
            # Enforce strict contract - result must be OperationResult
            if isinstance(result, OperationResult):
                success_status = result.success
            else:
                # Non-OperationResult indicates worker-level success
                success_status = True
            self._on_operation_finished(operation_name, success_status, result)
        
        def on_error(error):
            logger.info(f"Operation {operation_name} error: {error}")
            self._on_operation_finished(operation_name, False, error)
        
        def on_progress(progress):
            self.progress_updated.emit(operation_name, progress)
        
        # Use Qt.QueuedConnection for cross-thread signals
        from PySide6.QtCore import Qt
        worker.finished.connect(on_finished, Qt.QueuedConnection)
        worker.error.connect(on_error, Qt.QueuedConnection)
        worker.progress.connect(on_progress, Qt.QueuedConnection)
        
        self._current_workers[operation_name] = worker
        worker.start()
        
        logger.debug(f"Started operation: {operation_name}")
    
    def _on_operation_finished(self, operation_name: str, success: bool, result: Union[str, int, float, bool, dict, list, object], performance_monitor=None) -> None:
        """Handle operation completion with automatic OperationResult extraction."""
        logger.info(f"Operation finished: {operation_name}, success={success}, result_type={type(result)}")
        
        # Clean up worker
        if operation_name in self._current_workers:
            worker = self._current_workers[operation_name]
            worker.quit()
            worker.wait()
            del self._current_workers[operation_name]
            logger.info(f"Cleaned up worker for operation: {operation_name}")
        
        # Update state
        self.state.current_operations = max(0, self.state.current_operations - 1)
        
        if success:
            # Automatically extract data from OperationResult objects
            extracted_result = self._extract_operation_data(result)
            logger.info(f"Operation completed successfully: {operation_name}, extracted_result_type={type(extracted_result)}")
            self.operation_completed.emit(operation_name, True, extracted_result)
        else:
            logger.error(f"Operation failed: {operation_name} - {result}")
            self.operation_failed.emit(operation_name, str(result))
    
    def _extract_operation_data(self, result: Union[str, int, float, bool, dict, list, object]) -> Union[str, int, float, bool, dict, list]:
        """Extract actual data from OperationResult objects following CLAUDE.md fail-fast principles."""
        # Handle OperationResult objects by extracting their data - enforce strict contract
        if isinstance(result, OperationResult):
            # Always return the data from OperationResult - success status is handled separately
            data = result.data if result.data is not None else {}
            # Convert Pydantic models to dictionaries for UI consumption
            if hasattr(data, 'model_dump'):
                return data.model_dump()
            # Handle lists of Pydantic models
            elif isinstance(data, list) and data and hasattr(data[0], 'model_dump'):
                return [item.model_dump() for item in data]
            return data
        
        # For non-OperationResult objects, handle Pydantic conversion
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        elif isinstance(result, list) and result and hasattr(result[0], 'model_dump'):
            return [item.model_dump() for item in result]
        
        return result
    
    def cancel_operation(self, operation_name: str) -> None:
        """Cancel a running operation."""
        if operation_name in self._current_workers:
            worker = self._current_workers[operation_name]
            worker.quit()
            worker.wait()
            del self._current_workers[operation_name]
            self.state.current_operations = max(0, self.state.current_operations - 1)
            logger.debug(f"Operation cancelled: {operation_name}")
    
    async def shutdown(self) -> None:
        """Shutdown controller and cleanup resources."""
        try:
            logger.debug(f"Shutting down {self.__class__.__name__}")
            
            # Cancel all running operations
            for operation_name in list(self._current_workers.keys()):
                self.cancel_operation(operation_name)
            
            # Reset state
            self.state = ControllerState()
            
            logger.debug(f"{self.__class__.__name__} shutdown completed")
            
        except Exception as e:
            logger.error(f"Controller shutdown failed: {e}")
            raise
    
    def get_controller_state(self) -> dict:
        """Get current controller state."""
        return {
            "initialized": self.state.initialized,
            "current_operations": self.state.current_operations,
            "last_operation": self.state.last_operation,
            "active_workers": len(self._current_workers)
        }


class OperationWorker(QThread):
    """Worker thread for executing operations with async support."""
    
    finished = Signal(object)  # result
    error = Signal(str)  # error message
    progress = Signal(int)  # progress percentage
    
    def __init__(self, operation_func: Callable, *args, **kwargs):
        super().__init__()
        self.operation_func = operation_func
        self.args = args
        self.kwargs = kwargs
    
    def run(self) -> None:
        """Execute operation in worker thread."""
        try:
            # Execute operation function
            result = self.operation_func(*self.args, **self.kwargs)
            
            # Handle async operations
            if asyncio.iscoroutine(result):
                try:
                    # Create new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        async_result = loop.run_until_complete(result)
                        logger.info(f"Async operation completed, result: {async_result}")
                        self.finished.emit(async_result)
                        return
                    finally:
                        loop.close()
                        
                except Exception as e:
                    logger.error(f"Async operation execution failed: {e}")
                    self.error.emit(f"Async operation failed: {e}")
                    return
            
            # For sync operations
            logger.info(f"Sync operation completed")
            self.finished.emit(result)
            
        except Exception as e:
            logger.exception(f"Operation worker error: {e}")
            self.error.emit(str(e))


class ControllerManagerState(MutableStrictBaseModel):
    """Controller manager state with strict validation but mutable for runtime updates."""
    # Inherits strict configuration from MutableStrictBaseModel
    
    initialized: bool = False
    controller_count: int = 0


class ControllerManager(QObject):
    """
    Manages controller lifecycle with async-first design.
    
    No fallbacks, strict contracts only.
    """
    
    def __init__(self, service_factory):
        super().__init__()
        self.service_factory = service_factory
        self.state = ControllerManagerState()
        self._controllers = {}
        
        logger.info("ControllerManager initialized")
    
    async def get_controller(self, controller_class, controller_name: Optional[str] = None):
        """Get or create a controller instance."""
        if controller_name is None:
            controller_name = controller_class.__name__
            
        if controller_name not in self._controllers:
            try:
                controller = controller_class(self.service_factory)
                await controller.initialize()
                self._controllers[controller_name] = controller
                self.state.controller_count += 1
                logger.info(f"Controller created: {controller_name}")
            except Exception as e:
                logger.error(f"Failed to create controller {controller_name}: {e}")
                raise
                
        return self._controllers[controller_name]
    
    def get_controller_sync(self, controller_class, controller_name: Optional[str] = None):
        """Synchronous fallback for UI compatibility - creates controller without full initialization."""
        if controller_name is None:
            controller_name = controller_class.__name__
            
        if controller_name not in self._controllers:
            try:
                # Create controller but defer initialization
                controller = controller_class(self.service_factory)
                self._controllers[controller_name] = controller
                self.state.controller_count += 1
                logger.info(f"Controller created (sync): {controller_name} - initialization deferred")
            except Exception as e:
                logger.error(f"Failed to create controller {controller_name}: {e}")
                raise
                
        return self._controllers[controller_name]
    
    async def shutdown_all(self) -> None:
        """Shutdown all controllers."""
        try:
            logger.info("Shutting down all controllers")
            
            for controller_name, controller in self._controllers.items():
                try:
                    await controller.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down controller {controller_name}: {e}")
            
            self._controllers.clear()
            self.state = ControllerManagerState()
            logger.info("Controller shutdown completed")
            
        except Exception as e:
            logger.error(f"Controller manager shutdown failed: {e}")
            raise
    
    def get_manager_state(self) -> dict:
        """Get current manager state."""
        return {
            "initialized": self.state.initialized,
            "controller_count": self.state.controller_count,
            "active_controllers": len(self._controllers)
        }
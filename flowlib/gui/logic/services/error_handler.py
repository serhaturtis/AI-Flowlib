"""
Error handling utilities for the MVC architecture.

Provides standardized error handling and user feedback mechanisms.
"""

import logging
from typing import Optional, Callable
from PySide6.QtWidgets import QMessageBox, QWidget
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class ErrorHandler(QObject):
    """
    Centralized error handling for the application.
    
    Provides standardized error reporting and user feedback.
    """
    
    # Error signals
    critical_error = Signal(str, str)  # title, message
    warning_error = Signal(str, str)   # title, message
    info_message = Signal(str, str)    # title, message
    
    def __init__(self, parent_widget: Optional[QWidget] = None):
        super().__init__()
        self.parent_widget = parent_widget
        
        # Connect signals to show dialogs
        self.critical_error.connect(self._show_critical_dialog)
        self.warning_error.connect(self._show_warning_dialog)
        self.info_message.connect(self._show_info_dialog)
    
    def handle_service_error(self, service_name: str, error_message: str, critical: bool = False):
        """Handle service-related errors."""
        title = f"Service Error: {service_name}"
        message = f"The '{service_name}' service encountered an error:\n\n{error_message}"
        
        if critical:
            message += "\n\nThis may affect application functionality."
            self.critical_error.emit(title, message)
        else:
            message += "\n\nSome features may not be available."
            self.warning_error.emit(title, message)
        
        logger.error(f"Service error - {service_name}: {error_message}")
    
    def handle_controller_error(self, controller_name: str, operation: str, error_message: str):
        """Handle controller operation errors."""
        title = f"Operation Failed"
        message = f"The '{operation}' operation in {controller_name} failed:\n\n{error_message}"
        
        self.warning_error.emit(title, message)
        logger.error(f"Controller error - {controller_name}.{operation}: {error_message}")
    
    def handle_ui_error(self, component_name: str, error_message: str):
        """Handle UI-related errors."""
        title = "Interface Error"
        message = f"An error occurred in {component_name}:\n\n{error_message}"
        
        self.warning_error.emit(title, message)
        logger.error(f"UI error - {component_name}: {error_message}")
    
    def show_info(self, title: str, message: str):
        """Show an informational message."""
        self.info_message.emit(title, message)
        logger.info(f"Info message - {title}: {message}")
    
    def _show_critical_dialog(self, title: str, message: str):
        """Show critical error dialog."""
        QMessageBox.critical(self.parent_widget, title, message)
    
    def _show_warning_dialog(self, title: str, message: str):
        """Show warning dialog."""
        QMessageBox.warning(self.parent_widget, title, message)
    
    def _show_info_dialog(self, title: str, message: str):
        """Show info dialog."""
        QMessageBox.information(self.parent_widget, title, message)


def safe_operation(error_handler: ErrorHandler, operation_name: str, component_name: str):
    """
    Decorator for safe operation execution with error handling.
    
    Args:
        error_handler: ErrorHandler instance
        operation_name: Name of the operation being performed
        component_name: Name of the component performing the operation
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_controller_error(component_name, operation_name, str(e))
                return None
        return wrapper
    return decorator


class ServiceAvailabilityChecker:
    """
    Utility class for checking service availability and providing user feedback.
    """
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
    
    def check_service_availability(self, controller, service_name: str, operation_name: str) -> bool:
        """
        Check if a service is available before performing an operation.
        
        Args:
            controller: Controller instance to check
            service_name: Human-readable service name
            operation_name: Name of the operation being attempted
            
        Returns:
            True if service is available, False otherwise
        """
        if not controller:
            self.error_handler.handle_service_error(
                service_name,
                f"Controller not initialized for {operation_name}",
                critical=True
            )
            return False
        
        if not controller.is_service_available():
            self.error_handler.handle_service_error(
                service_name,
                f"Service not available for {operation_name}. Please check the configuration.",
                critical=False
            )
            return False
        
        return True
    
    def require_service(self, controller, service_name: str, operation_name: str) -> bool:
        """
        Require a service to be available, showing appropriate errors if not.
        
        This is a stricter version of check_service_availability that treats
        unavailable services as critical errors.
        """
        if not self.check_service_availability(controller, service_name, operation_name):
            self.error_handler.handle_service_error(
                service_name,
                f"Cannot perform {operation_name} - required service is not available.",
                critical=True
            )
            return False
        
        return True
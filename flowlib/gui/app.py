"""
Main GUI Application Entry Point.

Initializes the MVC architecture and provides the main application controller.
"""

import sys
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QObject, Signal

from .logic.services.service_factory import ServiceFactory
from .logic.services.base_controller import ControllerManager
from .ui.main_window import MainWindow

logger = logging.getLogger(__name__)


class ApplicationController(QObject):
    """
    Main application controller that manages the overall application lifecycle
    and coordinates between the UI and business logic layers.
    """
    
    # Application-level signals
    application_ready = Signal()
    application_error = Signal(str)
    services_initialized = Signal()
    
    def __init__(self):
        super().__init__()
        
        # Core MVC components
        self.service_factory = None
        self.controller_manager = None
        self.main_window = None
        
        # Application state
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the application and MVC architecture."""
        try:
            logger.info("Initializing GUI Configuration Application")
            
            # Initialize service factory
            self.service_factory = ServiceFactory()
            logger.info("Service factory initialized")
            
            # Initialize controller manager
            self.controller_manager = ControllerManager(self.service_factory)
            logger.info("Controller manager initialized")
            
            # Create main window
            self.main_window = MainWindow()
            logger.info("Main window created")
            
            # Connect application-level signals
            self.service_factory.service_failed.connect(self.on_service_error)
            
            self.is_initialized = True
            self.services_initialized.emit()
            
            logger.info("Application initialization completed successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize application: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.application_error.emit(error_msg)
            raise
    
    def show(self):
        """Show the main application window."""
        if not self.is_initialized:
            raise RuntimeError("Application must be initialized before showing")
        
        self.main_window.show()
        self.application_ready.emit()
        logger.info("Application window shown")
    
    def on_service_error(self, service_name: str, error_message: str):
        """Handle service initialization errors."""
        error_msg = f"Service '{service_name}' failed to initialize: {error_message}"
        logger.error(error_msg)
        
        # Show error dialog to user
        if self.main_window:
            QMessageBox.critical(
                self.main_window,
                "Service Error",
                f"A critical service failed to start:\n\n{error_msg}\n\n"
                "Some functionality may not be available."
            )
    
    def get_service_factory(self):
        """Get the service factory instance."""
        return self.service_factory
    
    def get_controller_manager(self):
        """Get the controller manager instance."""
        return self.controller_manager
    
    def get_main_window(self):
        """Get the main window instance."""
        return self.main_window
    
    def cleanup(self):
        """Clean up application resources."""
        logger.info("Cleaning up application resources")
        
        if self.controller_manager:
            self.controller_manager.cleanup_all()
        
        if self.service_factory:
            self.service_factory.cleanup()
        
        logger.info("Application cleanup completed")


def setup_logging():
    """Set up application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Flowlib Configuration Manager")
    app.setApplicationVersion("1.0.0")
    
    # Create and initialize application controller
    app_controller = ApplicationController()
    
    try:
        app_controller.initialize()
        app_controller.show()
        
        # Run the application
        exit_code = app.exec()
        
        # Clean up
        app_controller.cleanup()
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Critical application error: {e}", exc_info=True)
        
        # Try to show error dialog
        try:
            QMessageBox.critical(
                None,
                "Critical Error",
                f"The application encountered a critical error and cannot continue:\n\n{str(e)}"
            )
        except Exception as dialog_error:
            # If even the error dialog fails, log to stderr and exit
            print(f"CRITICAL: Could not display error dialog: {dialog_error}", file=sys.stderr)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
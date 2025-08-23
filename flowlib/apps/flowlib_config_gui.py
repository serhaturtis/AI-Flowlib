#!/usr/bin/env python3
"""
Flowlib GUI Configuration Manager

Qt-based graphical interface for managing Flowlib configurations.
Provides visual configuration management, provider setup, and knowledge plugin management.
"""

import sys
import os
import logging
import multiprocessing
import faulthandler
from pathlib import Path
from datetime import datetime

# Add flowlib to path for imports (accounting for new structure)
flowlib_root = Path(__file__).parent.parent  # Go up from apps/ to flowlib/
sys.path.insert(0, str(flowlib_root.parent))  # Add AI-Flowlib root to path

def setup_logging():
    """Set up logging for the GUI application."""
    log_dir = Path.home() / '.flowlib' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'gui.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main entry point for the Flowlib Configuration GUI."""
    # Enable fault handler for better debugging
    faulthandler.enable()
    multiprocessing.freeze_support()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Import Qt components
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QIcon
        
        # Import flowlib GUI components with proper paths
        from flowlib.gui.ui.main_window import MainWindow
        from flowlib.gui.ui.style.style_manager import StyleManager
        from flowlib.gui import config
        
        logger.info("Successfully imported GUI components")
        
    except ImportError as e:
        error_msg = f"""
❌ GUI IMPORT ERROR
Failed to import GUI components: {e}

Required dependencies:
- PySide6: pip install PySide6
- Flowlib GUI modules must be present in flowlib/gui/

To install PySide6:
    pip install PySide6

To check GUI module availability:
    python -c "from flowlib.gui import config; print('GUI modules available')"
"""
        print(error_msg)
        logger.error(f"Import error: {e}")
        return 1
    
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName(config.APP_NAME)
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("AI-Flowlib")
        
        # Set application icon if available
        try:
            if os.path.exists(config.APP_LOGO_PNG_PATH):
                app.setWindowIcon(QIcon(config.APP_LOGO_PNG_PATH))
                logger.info(f"Set application icon: {config.APP_LOGO_PNG_PATH}")
            elif os.path.exists(config.APP_LOGO_ICO_PATH):
                app.setWindowIcon(QIcon(config.APP_LOGO_ICO_PATH))
                logger.info(f"Set application icon: {config.APP_LOGO_ICO_PATH}")
        except Exception as e:
            logger.warning(f"Could not set application icon: {e}")
        
        # Apply modern styling/theme
        try:
            StyleManager.apply_stylesheet(app, theme="theme.xml")
            logger.info("Applied application theme successfully")
        except Exception as e:
            logger.warning(f"Failed to apply theme: {e}")
            # Continue without styling
        
        # Handle license agreement (auto-accept for development)
        try:
            if not os.path.exists(config.LCS_FILE):
                os.makedirs(os.path.dirname(config.LCS_FILE), exist_ok=True)
                with open(config.LCS_FILE, 'w') as f:
                    f.write(str(datetime.now()))
                logger.info("License acceptance file created")
        except Exception as e:
            logger.warning(f"License file handling failed: {e}")
            # Continue without license file
        
        logger.info("Starting Flowlib Configuration GUI")
        
        # Create and show main window
        window = MainWindow()
        window.setWindowTitle(f"{config.APP_NAME} - Configuration Manager")
        window.resize(1280, 800)
        window.show()
        
        logger.info("GUI window created and displayed")
        
        # Run application event loop
        return app.exec()
        
    except Exception as e:
        error_msg = f"""
❌ CONFIGURATION GUI ERROR
An unexpected error occurred while running the GUI.

Error details: {e}

Logs are saved to: ~/.flowlib/logs/gui.log

Common solutions:
1. Ensure PySide6 is properly installed: pip install PySide6
2. Check that flowlib.gui modules are available
3. Verify Qt libraries are properly installed on your system
"""
        print(error_msg)
        
        # Log detailed error information
        logger.error(f"GUI runtime error: {e}", exc_info=True)
        
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Configuration GUI closed by user")
        sys.exit(0)
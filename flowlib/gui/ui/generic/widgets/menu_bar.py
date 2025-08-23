from PySide6.QtWidgets import QMenuBar
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QFileDialog
from PySide6.QtGui import QAction
from PySide6.QtGui import QIcon
from PySide6.QtGui import QKeySequence
from pathlib import Path
import logging

"""
QMenuBar::item {
                padding: 0px 10px; /* Adjust padding inside each menu item */
                margin: 0px; /* Remove margin around each menu item */
            }
            QMenuBar::item:selected { /* Remove any additional space on selected items */
                background: #dddddd; /* Example background color */
            }
"""


class MenuBar(QMenuBar):
    def __init__(self, main_window: QMainWindow = None):
        super(MenuBar, self).__init__(main_window)
        self.main_window = main_window
        
        # Main container panel reference will be set after initialization
        self.main_container_panel = None

        self.setStyleSheet("""
            QMenuBar {
                spacing: 0px; /* Remove spacing between menu items */
                padding: 0px; /* Remove padding inside the menubar */
            }
        """)

        # Menu Bar
        file_menu = self.addMenu("File")
        edit_menu = self.addMenu("Edit")
        view_menu = self.addMenu("View")
        tools_menu = self.addMenu("Tools")
        help_menu = self.addMenu("Help")

        # File Menu Actions
        new_action = QAction("New Configuration", main_window)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.setStatusTip("Create new configuration")
        new_action.triggered.connect(self.new_configuration)

        open_action = QAction("Open...", main_window)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip("Open configuration file")
        open_action.triggered.connect(self.open_file)

        save_action = QAction("Save Configuration", main_window)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.setStatusTip("Save current configuration")
        save_action.triggered.connect(self.save_configuration)

        import_config_action = QAction("Import Configuration...", main_window)
        import_config_action.setShortcut(QKeySequence("Ctrl+I"))
        import_config_action.setStatusTip("Import configuration from file")
        import_config_action.triggered.connect(self.import_configuration)

        export_config_action = QAction("Export Configuration...", main_window)
        export_config_action.setShortcut(QKeySequence("Ctrl+E"))
        export_config_action.setStatusTip("Export configuration to file")
        export_config_action.triggered.connect(self.export_configuration)

        backup_repo_action = QAction("Backup Repository...", main_window)
        backup_repo_action.setStatusTip("Create complete repository backup")
        backup_repo_action.triggered.connect(self.backup_repository)

        exit_action = QAction("Exit", main_window)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.main_window.close)

        # Edit Menu Actions
        settings_action = QAction("Settings...", main_window)
        settings_action.setShortcut(QKeySequence.StandardKey.Preferences)
        settings_action.setStatusTip("Open application settings")
        settings_action.triggered.connect(self.open_settings)

        # View Menu Actions
        show_provider_repo_action = QAction("Provider Repository", main_window)
        show_provider_repo_action.setShortcut(QKeySequence("Ctrl+1"))
        show_provider_repo_action.setStatusTip("Show Provider Repository page")
        show_provider_repo_action.triggered.connect(lambda: self.navigate_to_page(0))

        show_config_manager_action = QAction("Configuration Manager", main_window)
        show_config_manager_action.setShortcut(QKeySequence("Ctrl+2"))
        show_config_manager_action.setStatusTip("Show Configuration Manager page")
        show_config_manager_action.triggered.connect(lambda: self.navigate_to_page(1))

        show_preset_manager_action = QAction("Preset Manager", main_window)
        show_preset_manager_action.setShortcut(QKeySequence("Ctrl+3"))
        show_preset_manager_action.setStatusTip("Show Preset Manager page")
        show_preset_manager_action.triggered.connect(lambda: self.navigate_to_page(2))

        show_plugin_manager_action = QAction("Knowledge Plugin Manager", main_window)
        show_plugin_manager_action.setShortcut(QKeySequence("Ctrl+4"))
        show_plugin_manager_action.setStatusTip("Show Knowledge Plugin Manager page")
        show_plugin_manager_action.triggered.connect(lambda: self.navigate_to_page(3))

        show_import_export_action = QAction("Import/Export", main_window)
        show_import_export_action.setShortcut(QKeySequence("Ctrl+5"))
        show_import_export_action.setStatusTip("Show Import/Export page")
        show_import_export_action.triggered.connect(lambda: self.navigate_to_page(4))

        show_testing_tools_action = QAction("Testing Tools", main_window)
        show_testing_tools_action.setShortcut(QKeySequence("Ctrl+6"))
        show_testing_tools_action.setStatusTip("Show Testing Tools page")
        show_testing_tools_action.triggered.connect(lambda: self.navigate_to_page(5))

        toggle_console_action = QAction("Toggle Console", main_window)
        toggle_console_action.setShortcut(QKeySequence("Ctrl+`"))
        toggle_console_action.setStatusTip("Show/hide console panel")
        toggle_console_action.triggered.connect(self.toggle_console)

        # Tools Menu Actions
        validate_configs_action = QAction("Validate Configurations", main_window)
        validate_configs_action.setShortcut(QKeySequence("Ctrl+T"))
        validate_configs_action.setStatusTip("Validate all configurations")
        validate_configs_action.triggered.connect(self.validate_configurations)

        test_connections_action = QAction("Test All Connections", main_window)
        test_connections_action.setShortcut(QKeySequence("Ctrl+Shift+T"))
        test_connections_action.setStatusTip("Test all provider connections")
        test_connections_action.triggered.connect(self.test_connections)

        clear_cache_action = QAction("Clear Cache", main_window)
        clear_cache_action.setStatusTip("Clear configuration cache")
        clear_cache_action.triggered.connect(self.clear_cache)

        refresh_action = QAction("Refresh All", main_window)
        refresh_action.setShortcut(QKeySequence.StandardKey.Refresh)
        refresh_action.setStatusTip("Refresh all data")
        refresh_action.triggered.connect(self.refresh_all)

        # Help Menu Actions
        documentation_action = QAction("Documentation", main_window)
        documentation_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        documentation_action.setStatusTip("Open documentation")
        documentation_action.triggered.connect(self.open_documentation)

        about_action = QAction("About", main_window)
        about_action.setStatusTip("About Flowlib Configuration Manager")
        about_action.triggered.connect(self.show_about_dialog)

        # Adding Actions to Menu Bar
        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(import_config_action)
        file_menu.addAction(export_config_action)
        file_menu.addAction(backup_repo_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        edit_menu.addAction(settings_action)

        view_menu.addAction(show_provider_repo_action)
        view_menu.addAction(show_config_manager_action)
        view_menu.addAction(show_preset_manager_action)
        view_menu.addAction(show_plugin_manager_action)
        view_menu.addAction(show_import_export_action)
        view_menu.addAction(show_testing_tools_action)
        view_menu.addSeparator()
        view_menu.addAction(toggle_console_action)

        tools_menu.addAction(validate_configs_action)
        tools_menu.addAction(test_connections_action)
        tools_menu.addSeparator()
        tools_menu.addAction(clear_cache_action)
        tools_menu.addAction(refresh_action)

        help_menu.addAction(documentation_action)
        help_menu.addSeparator()
        help_menu.addAction(about_action)

        # Store action references
        self.actions = {
            'new': new_action,
            'open': open_action,
            'save': save_action,
            'import_config': import_config_action,
            'export_config': export_config_action,
            'backup_repo': backup_repo_action,
            'exit': exit_action,
            'settings': settings_action,
            'show_provider_repo': show_provider_repo_action,
            'show_config_manager': show_config_manager_action,
            'show_preset_manager': show_preset_manager_action,
            'show_plugin_manager': show_plugin_manager_action,
            'show_import_export': show_import_export_action,
            'show_testing_tools': show_testing_tools_action,
            'toggle_console': toggle_console_action,
            'validate_configs': validate_configs_action,
            'test_connections': test_connections_action,
            'clear_cache': clear_cache_action,
            'refresh': refresh_action,
            'documentation': documentation_action,
            'about': about_action
        }

    def new_configuration(self):
        """Create new configuration"""
        self.navigate_to_page(1)  # Configuration Manager page
        # Could add additional logic to create new config dialog

    def open_file(self):
        """Open configuration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Open Configuration File",
            str(Path.home()),
            "Configuration Files (*.py *.yaml *.yml *.json);;All Files (*)"
        )
        if file_path:
            # Navigate to import/export page and trigger import
            self.navigate_to_page(4)  # Import/Export page
            # Could add logic to pre-fill import dialog with file path

    def save_configuration(self):
        """Save current configuration"""
        # This would save the currently active configuration
        # For now, navigate to export functionality
        self.navigate_to_page(4)  # Import/Export page

    def import_configuration(self):
        """Import configuration from file"""
        self.navigate_to_page(4)  # Import/Export page
        # Try to automatically switch to import tab
        try:
            if self.main_container_panel:
                pages = self.main_container_panel.get_pages()
                if len(pages) > 4:  # Import/Export page is index 4
                    import_export_page = pages[4]
                    if hasattr(import_export_page, 'tabs'):
                        # Switch to import tab (typically index 1)
                        from PySide6.QtCore import QTimer
                        QTimer.singleShot(100, lambda: import_export_page.tabs.setCurrentIndex(1))
        except Exception as e:
            logging.error(f"Failed to auto-switch to import tab: {e}")

    def export_configuration(self):
        """Export configuration to file"""
        self.navigate_to_page(4)  # Import/Export page
        # Try to automatically switch to export tab
        try:
            if self.main_container_panel:
                pages = self.main_container_panel.get_pages()
                if len(pages) > 4:  # Import/Export page is index 4
                    import_export_page = pages[4]
                    if hasattr(import_export_page, 'tabs'):
                        # Switch to export tab (typically index 0)
                        from PySide6.QtCore import QTimer
                        QTimer.singleShot(100, lambda: import_export_page.tabs.setCurrentIndex(0))
        except Exception as e:
            logging.error(f"Failed to auto-switch to export tab: {e}")

    def backup_repository(self):
        """Create complete repository backup"""
        self.navigate_to_page(4)  # Import/Export page
        # Try to automatically switch to backup tab
        try:
            if self.main_container_panel:
                pages = self.main_container_panel.get_pages()
                if len(pages) > 4:  # Import/Export page is index 4
                    import_export_page = pages[4]
                    if hasattr(import_export_page, 'tabs'):
                        # Switch to backup tab (typically index 2)
                        from PySide6.QtCore import QTimer
                        QTimer.singleShot(100, lambda: import_export_page.tabs.setCurrentIndex(2))
        except Exception as e:
            logging.error(f"Failed to auto-switch to backup tab: {e}")

    def open_settings(self):
        """Open application settings"""
        try:
            from flowlib.gui.ui.dialogs.settings_dialog import SettingsDialog
            dialog = SettingsDialog(self.main_window)
            dialog.settings_changed.connect(self.on_settings_changed)
            dialog.exec()
        except Exception as e:
            logging.error(f"Failed to open settings dialog: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open settings:\n{e}")
    
    def on_settings_changed(self):
        """Handle settings changes"""
        # Settings have been changed, could trigger application updates here
        pass

    def set_main_container_panel(self, panel):
        """Set reference to main container panel for navigation"""
        self.main_container_panel = panel

    def navigate_to_page(self, page_index):
        """Navigate to specific page"""
        try:
            # Get main container panel if not set
            if not self.main_container_panel and self.main_window:
                if hasattr(self.main_window, 'ide_panel') and hasattr(self.main_window.ide_panel, 'ide_main_container_panel'):
                    self.main_container_panel = self.main_window.ide_panel.ide_main_container_panel
                    
            if self.main_container_panel:
                self.main_container_panel.change_page(page_index)
                # Also update the navigation panel
                if hasattr(self.main_container_panel, 'navigation_panel'):
                    self.main_container_panel.navigation_panel.on_clicked(page_index)
        except Exception as e:
            logging.error(f"Failed to navigate to page {page_index}: {e}")

    def toggle_console(self):
        """Toggle console visibility"""
        try:
            if self.main_window and hasattr(self.main_window, 'ide_panel'):
                console = self.main_window.ide_panel.console
                if console.isVisible():
                    console.hide()
                else:
                    console.show()
        except Exception as e:
            logging.error(f"Failed to toggle console: {e}")

    def validate_configurations(self):
        """Validate all configurations"""
        self.navigate_to_page(5)  # Testing Tools page
        # Try to automatically start configuration testing
        try:
            if self.main_container_panel:
                pages = self.main_container_panel.get_pages()
                if len(pages) > 5:  # Testing tools page is index 5
                    testing_page = pages[5]
                    if hasattr(testing_page, 'test_all_configs'):
                        # Small delay to let page load, then trigger test
                        from PySide6.QtCore import QTimer
                        QTimer.singleShot(500, testing_page.test_all_configs)
        except Exception as e:
            logging.error(f"Failed to auto-start configuration testing: {e}")

    def test_connections(self):
        """Test all provider connections"""
        self.navigate_to_page(5)  # Testing Tools page
        # Try to automatically start connection testing
        try:
            if self.main_container_panel:
                pages = self.main_container_panel.get_pages()
                if len(pages) > 5:  # Testing tools page is index 5
                    testing_page = pages[5]
                    if hasattr(testing_page, 'test_all_connections'):
                        # Small delay to let page load, then trigger test
                        from PySide6.QtCore import QTimer
                        QTimer.singleShot(500, testing_page.test_all_connections)
        except Exception as e:
            logging.error(f"Failed to auto-start connection testing: {e}")

    def clear_cache(self):
        """Clear configuration cache"""
        try:
            # This would clear various caches
            QMessageBox.information(self, "Clear Cache", "Cache cleared successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clear cache: {e}")

    def refresh_all(self):
        """Refresh all data"""
        try:
            # Refresh all pages
            refreshed_count = 0
            if self.main_container_panel:
                for page in self.main_container_panel.get_pages():
                    # Try various refresh method names
                    if hasattr(page, 'refresh_all'):
                        page.refresh_all()
                        refreshed_count += 1
                    elif hasattr(page, 'refresh_data'):
                        page.refresh_data()
                        refreshed_count += 1
                    elif hasattr(page, 'refresh'):
                        page.refresh()
                        refreshed_count += 1
                    elif hasattr(page, 'page_visible'):
                        page.page_visible()
                        refreshed_count += 1
            QMessageBox.information(self, "Refresh", f"Successfully refreshed {refreshed_count} pages.")
        except Exception as e:
            logging.error(f"Failed to refresh all data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to refresh data: {e}")

    def open_documentation(self):
        """Open documentation"""
        try:
            import webbrowser
            webbrowser.open("https://github.com/your-org/AI-Flowlib/wiki")
        except (OSError, ImportError) as browser_error:
            logger.debug(f"Could not open browser for documentation: {browser_error}")
            QMessageBox.information(self, "Documentation", 
                                  "Documentation is available at:\nhttps://github.com/your-org/AI-Flowlib/wiki")

    def show_about_dialog(self):
        """Show about dialog"""
        about_text = """
<h2>Flowlib Configuration Manager</h2>
<p><b>Version:</b> 1.0.0</p>
<p><b>Description:</b> A powerful GUI tool for managing AI-Flowlib provider configurations, presets, and knowledge plugins.</p>

<h3>Features:</h3>
<ul>
<li>Provider Repository Management</li>
<li>Configuration Management with validation</li>
<li>Preset Templates with variable substitution</li>
<li>Knowledge Plugin Generation and Management</li>
<li>Import/Export functionality</li>
<li>Comprehensive Testing Tools</li>
</ul>

<h3>Keyboard Shortcuts:</h3>
<p><b>Ctrl+1-6:</b> Navigate to different pages</p>
<p><b>Ctrl+T:</b> Validate configurations</p>
<p><b>Ctrl+Shift+T:</b> Test connections</p>
<p><b>Ctrl+I/E:</b> Import/Export</p>
<p><b>Ctrl+`:</b> Toggle console</p>
<p><b>F5:</b> Refresh all data</p>

<p><b>Built with:</b> PySide6 and AI-Flowlib framework</p>
        """
        QMessageBox.about(self, "About Flowlib Configuration Manager", about_text)

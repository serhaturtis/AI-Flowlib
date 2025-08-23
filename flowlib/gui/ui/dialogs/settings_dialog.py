from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, 
                               QLabel, QLineEdit, QCheckBox, QSpinBox, QComboBox,
                               QGroupBox, QFormLayout, QDialogButtonBox, QPushButton,
                               QFileDialog, QMessageBox, QTextEdit, QSlider)
from PySide6.QtCore import Qt, Signal
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Settings dialog for application configuration"""
    
    settings_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flowlib Configuration Manager Settings")
        self.setModal(True)
        self.resize(600, 500)
        
        # Settings storage
        self.settings = self.load_settings()
        
        self.init_ui()
        self.load_current_settings()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Create tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # General tab
        general_tab = self.create_general_tab()
        tabs.addTab(general_tab, "General")
        
        # Paths tab
        paths_tab = self.create_paths_tab()
        tabs.addTab(paths_tab, "Paths")
        
        # Interface tab
        interface_tab = self.create_interface_tab()
        tabs.addTab(interface_tab, "Interface")
        
        # Advanced tab
        advanced_tab = self.create_advanced_tab()
        tabs.addTab(advanced_tab, "Advanced")
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel | 
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        button_box.accepted.connect(self.accept_settings)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_settings)
        button_box.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self.restore_defaults)
        layout.addWidget(button_box)
    
    def create_general_tab(self):
        """Create general settings tab"""
        tab = QTabWidget()
        layout = QVBoxLayout(tab)
        
        # Application group
        app_group = QGroupBox("Application Settings")
        app_layout = QFormLayout(app_group)
        layout.addWidget(app_group)
        
        self.auto_save_check = QCheckBox("Auto-save configurations")
        app_layout.addRow("Auto-save:", self.auto_save_check)
        
        self.auto_save_interval = QSpinBox()
        self.auto_save_interval.setRange(1, 60)
        self.auto_save_interval.setSuffix(" minutes")
        app_layout.addRow("Auto-save interval:", self.auto_save_interval)
        
        self.check_updates_check = QCheckBox("Check for updates on startup")
        app_layout.addRow("Updates:", self.check_updates_check)
        
        self.backup_before_changes_check = QCheckBox("Create backup before making changes")
        app_layout.addRow("Backup:", self.backup_before_changes_check)
        
        # Validation group
        validation_group = QGroupBox("Validation Settings")
        validation_layout = QFormLayout(validation_group)
        layout.addWidget(validation_group)
        
        self.validate_on_save_check = QCheckBox("Validate configurations on save")
        validation_layout.addRow("Auto-validation:", self.validate_on_save_check)
        
        self.validation_timeout = QSpinBox()
        self.validation_timeout.setRange(5, 300)
        self.validation_timeout.setSuffix(" seconds")
        validation_layout.addRow("Validation timeout:", self.validation_timeout)
        
        layout.addStretch()
        return tab
    
    def create_paths_tab(self):
        """Create paths settings tab"""
        tab = QTabWidget()
        layout = QVBoxLayout(tab)
        
        # Paths group
        paths_group = QGroupBox("Directory Paths")
        paths_layout = QFormLayout(paths_group)
        layout.addWidget(paths_group)
        
        # Flowlib config directory
        config_layout = QHBoxLayout()
        self.config_dir_edit = QLineEdit()
        config_browse_btn = QPushButton("Browse...")
        config_browse_btn.clicked.connect(lambda: self.browse_directory(self.config_dir_edit))
        config_layout.addWidget(self.config_dir_edit)
        config_layout.addWidget(config_browse_btn)
        paths_layout.addRow("Config directory:", config_layout)
        
        # Plugin directory
        plugin_layout = QHBoxLayout()
        self.plugin_dir_edit = QLineEdit()
        plugin_browse_btn = QPushButton("Browse...")
        plugin_browse_btn.clicked.connect(lambda: self.browse_directory(self.plugin_dir_edit))
        plugin_layout.addWidget(self.plugin_dir_edit)
        plugin_layout.addWidget(plugin_browse_btn)
        paths_layout.addRow("Plugin directory:", plugin_layout)
        
        # Backup directory
        backup_layout = QHBoxLayout()
        self.backup_dir_edit = QLineEdit()
        backup_browse_btn = QPushButton("Browse...")
        backup_browse_btn.clicked.connect(lambda: self.browse_directory(self.backup_dir_edit))
        backup_layout.addWidget(self.backup_dir_edit)
        backup_layout.addWidget(backup_browse_btn)
        paths_layout.addRow("Backup directory:", backup_layout)
        
        # Temp directory
        temp_layout = QHBoxLayout()
        self.temp_dir_edit = QLineEdit()
        temp_browse_btn = QPushButton("Browse...")
        temp_browse_btn.clicked.connect(lambda: self.browse_directory(self.temp_dir_edit))
        temp_layout.addWidget(self.temp_dir_edit)
        temp_layout.addWidget(temp_browse_btn)
        paths_layout.addRow("Temp directory:", temp_layout)
        
        layout.addStretch()
        return tab
    
    def create_interface_tab(self):
        """Create interface settings tab"""
        tab = QTabWidget()
        layout = QVBoxLayout(tab)
        
        # Theme group
        theme_group = QGroupBox("Appearance")
        theme_layout = QFormLayout(theme_group)
        layout.addWidget(theme_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Auto"])
        theme_layout.addRow("Theme:", self.theme_combo)
        
        self.font_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_size_slider.setRange(8, 20)
        self.font_size_label = QLabel("12")
        font_size_layout = QHBoxLayout()
        font_size_layout.addWidget(self.font_size_slider)
        font_size_layout.addWidget(self.font_size_label)
        self.font_size_slider.valueChanged.connect(lambda v: self.font_size_label.setText(str(v)))
        theme_layout.addRow("Font size:", font_size_layout)
        
        # Console group
        console_group = QGroupBox("Console Settings")
        console_layout = QFormLayout(console_group)
        layout.addWidget(console_group)
        
        self.console_visible_check = QCheckBox("Show console by default")
        console_layout.addRow("Visibility:", self.console_visible_check)
        
        self.console_height = QSpinBox()
        self.console_height.setRange(50, 500)
        self.console_height.setSuffix(" pixels")
        console_layout.addRow("Console height:", self.console_height)
        
        self.max_console_lines = QSpinBox()
        self.max_console_lines.setRange(100, 10000)
        self.max_console_lines.setSuffix(" lines")
        console_layout.addRow("Max console lines:", self.max_console_lines)
        
        layout.addStretch()
        return tab
    
    def create_advanced_tab(self):
        """Create advanced settings tab"""
        tab = QTabWidget()
        layout = QVBoxLayout(tab)
        
        # Logging group
        logging_group = QGroupBox("Logging Settings")
        logging_layout = QFormLayout(logging_group)
        layout.addWidget(logging_group)
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        logging_layout.addRow("Log level:", self.log_level_combo)
        
        self.enable_file_logging_check = QCheckBox("Enable file logging")
        logging_layout.addRow("File logging:", self.enable_file_logging_check)
        
        # Performance group
        performance_group = QGroupBox("Performance Settings")
        performance_layout = QFormLayout(performance_group)
        layout.addWidget(performance_group)
        
        self.worker_threads = QSpinBox()
        self.worker_threads.setRange(1, 16)
        performance_layout.addRow("Worker threads:", self.worker_threads)
        
        self.cache_size = QSpinBox()
        self.cache_size.setRange(10, 1000)
        self.cache_size.setSuffix(" MB")
        performance_layout.addRow("Cache size:", self.cache_size)
        
        self.connection_timeout = QSpinBox()
        self.connection_timeout.setRange(5, 300)
        self.connection_timeout.setSuffix(" seconds")
        performance_layout.addRow("Connection timeout:", self.connection_timeout)
        
        # Debug group
        debug_group = QGroupBox("Debug Settings")
        debug_layout = QFormLayout(debug_group)
        layout.addWidget(debug_group)
        
        self.debug_mode_check = QCheckBox("Enable debug mode")
        debug_layout.addRow("Debug mode:", self.debug_mode_check)
        
        self.verbose_logging_check = QCheckBox("Verbose logging")
        debug_layout.addRow("Verbose:", self.verbose_logging_check)
        
        layout.addStretch()
        return tab
    
    def browse_directory(self, line_edit):
        """Browse for directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", line_edit.text() or str(Path.home())
        )
        if directory:
            line_edit.setText(directory)
    
    def load_settings(self):
        """Load settings from file"""
        settings_file = Path.home() / '.flowlib' / 'gui_settings.json'
        default_settings = {
            'general': {
                'auto_save': True,
                'auto_save_interval': 5,
                'check_updates': True,
                'backup_before_changes': True,
                'validate_on_save': True,
                'validation_timeout': 30
            },
            'paths': {
                'config_dir': str(Path.home() / '.flowlib'),
                'plugin_dir': str(Path.home() / '.flowlib' / 'knowledge_plugins'),
                'backup_dir': str(Path.home() / '.flowlib' / 'backups'),
                'temp_dir': str(Path.home() / '.flowlib' / 'temp')
            },
            'interface': {
                'theme': 'Dark',
                'font_size': 12,
                'console_visible': True,
                'console_height': 120,
                'max_console_lines': 1000
            },
            'advanced': {
                'log_level': 'INFO',
                'enable_file_logging': True,
                'worker_threads': 4,
                'cache_size': 100,
                'connection_timeout': 30,
                'debug_mode': False,
                'verbose_logging': False
            }
        }
        
        try:
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                # Merge with defaults to ensure all keys exist
                for category, values in default_settings.items():
                    if category not in loaded_settings:
                        loaded_settings[category] = {}
                    for key, default_value in values.items():
                        if key not in loaded_settings[category]:
                            loaded_settings[category][key] = default_value
                return loaded_settings
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
        
        return default_settings
    
    def save_settings(self):
        """Save settings to file"""
        try:
            settings_file = Path.home() / '.flowlib' / 'gui_settings.json'
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            logger.info("Settings saved successfully")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")
    
    def load_current_settings(self):
        """Load current settings into UI"""
        try:
            # General settings
            self.auto_save_check.setChecked(self.settings['general']['auto_save'])
            self.auto_save_interval.setValue(self.settings['general']['auto_save_interval'])
            self.check_updates_check.setChecked(self.settings['general']['check_updates'])
            self.backup_before_changes_check.setChecked(self.settings['general']['backup_before_changes'])
            self.validate_on_save_check.setChecked(self.settings['general']['validate_on_save'])
            self.validation_timeout.setValue(self.settings['general']['validation_timeout'])
            
            # Paths settings
            self.config_dir_edit.setText(self.settings['paths']['config_dir'])
            self.plugin_dir_edit.setText(self.settings['paths']['plugin_dir'])
            self.backup_dir_edit.setText(self.settings['paths']['backup_dir'])
            self.temp_dir_edit.setText(self.settings['paths']['temp_dir'])
            
            # Interface settings
            self.theme_combo.setCurrentText(self.settings['interface']['theme'])
            self.font_size_slider.setValue(self.settings['interface']['font_size'])
            self.font_size_label.setText(str(self.settings['interface']['font_size']))
            self.console_visible_check.setChecked(self.settings['interface']['console_visible'])
            self.console_height.setValue(self.settings['interface']['console_height'])
            self.max_console_lines.setValue(self.settings['interface']['max_console_lines'])
            
            # Advanced settings
            self.log_level_combo.setCurrentText(self.settings['advanced']['log_level'])
            self.enable_file_logging_check.setChecked(self.settings['advanced']['enable_file_logging'])
            self.worker_threads.setValue(self.settings['advanced']['worker_threads'])
            self.cache_size.setValue(self.settings['advanced']['cache_size'])
            self.connection_timeout.setValue(self.settings['advanced']['connection_timeout'])
            self.debug_mode_check.setChecked(self.settings['advanced']['debug_mode'])
            self.verbose_logging_check.setChecked(self.settings['advanced']['verbose_logging'])
            
        except Exception as e:
            logger.error(f"Failed to load current settings: {e}")
    
    def collect_settings(self):
        """Collect settings from UI"""
        try:
            self.settings = {
                'general': {
                    'auto_save': self.auto_save_check.isChecked(),
                    'auto_save_interval': self.auto_save_interval.value(),
                    'check_updates': self.check_updates_check.isChecked(),
                    'backup_before_changes': self.backup_before_changes_check.isChecked(),
                    'validate_on_save': self.validate_on_save_check.isChecked(),
                    'validation_timeout': self.validation_timeout.value()
                },
                'paths': {
                    'config_dir': self.config_dir_edit.text(),
                    'plugin_dir': self.plugin_dir_edit.text(),
                    'backup_dir': self.backup_dir_edit.text(),
                    'temp_dir': self.temp_dir_edit.text()
                },
                'interface': {
                    'theme': self.theme_combo.currentText(),
                    'font_size': self.font_size_slider.value(),
                    'console_visible': self.console_visible_check.isChecked(),
                    'console_height': self.console_height.value(),
                    'max_console_lines': self.max_console_lines.value()
                },
                'advanced': {
                    'log_level': self.log_level_combo.currentText(),
                    'enable_file_logging': self.enable_file_logging_check.isChecked(),
                    'worker_threads': self.worker_threads.value(),
                    'cache_size': self.cache_size.value(),
                    'connection_timeout': self.connection_timeout.value(),
                    'debug_mode': self.debug_mode_check.isChecked(),
                    'verbose_logging': self.verbose_logging_check.isChecked()
                }
            }
        except Exception as e:
            logger.error(f"Failed to collect settings: {e}")
    
    def apply_settings(self):
        """Apply settings without closing dialog"""
        self.collect_settings()
        self.save_settings()
        self.settings_changed.emit()
        QMessageBox.information(self, "Settings", "Settings applied successfully!")
    
    def accept_settings(self):
        """Accept and apply settings"""
        self.collect_settings()
        self.save_settings()
        self.settings_changed.emit()
        self.accept()
    
    def restore_defaults(self):
        """Restore default settings"""
        reply = QMessageBox.question(
            self, "Restore Defaults",
            "Are you sure you want to restore default settings?\nThis will overwrite all current settings.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Remove settings file to trigger default loading
            settings_file = Path.home() / '.flowlib' / 'gui_settings.json'
            try:
                if settings_file.exists():
                    settings_file.unlink()
                self.settings = self.load_settings()
                self.load_current_settings()
                QMessageBox.information(self, "Settings", "Default settings restored!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to restore defaults:\n{e}")
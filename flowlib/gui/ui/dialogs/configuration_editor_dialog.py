"""
Enhanced Configuration Editor Dialog

Provides advanced configuration editing with syntax highlighting,
real-time validation, and comprehensive error reporting.
"""

import json
import yaml
import logging
from typing import List, Optional, Union
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QTextEdit, QLabel,
    QPushButton, QTabWidget, QWidget, QListWidget, QListWidgetItem,
    QGroupBox, QGridLayout, QComboBox, QLineEdit, QCheckBox,
    QMessageBox, QFileDialog, QProgressBar, QTreeWidget, QTreeWidgetItem,
    QToolBar, QToolButton, QStatusBar, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtGui import QFont, QSyntaxHighlighter, QTextCharFormat, QColor, QAction

# Import template manager
try:
    from ..widgets.template_manager import TemplateManagerWidget
except ImportError:
    # Fallback if import fails
    TemplateManagerWidget = None

logger = logging.getLogger(__name__)


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python configuration files."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []
        
        # Keyword format
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(86, 156, 214))  # Blue
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            'from', 'import', 'class', 'def', 'return', 'if', 'else', 'elif',
            'for', 'while', 'try', 'except', 'finally', 'with', 'as', 'pass',
            'break', 'continue', 'and', 'or', 'not', 'in', 'is', 'None',
            'True', 'False', '@llm_config', '@model_config', '@database_config', '@vector_config'
        ]
        for keyword in keywords:
            pattern = f'\\b{keyword}\\b'
            self.highlighting_rules.append((pattern, keyword_format))
        
        # String format
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(206, 145, 120))  # Orange
        self.highlighting_rules.append(('\".*\"', string_format))
        self.highlighting_rules.append(("\'.*\'", string_format))
        
        # Comment format
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(106, 153, 85))  # Green
        self.highlighting_rules.append(('#.*', comment_format))
        
        # Decorator format
        decorator_format = QTextCharFormat()
        decorator_format.setForeground(QColor(220, 220, 170))  # Yellow
        self.highlighting_rules.append(('@\\w+', decorator_format))
    
    def highlightBlock(self, text):
        """Apply syntax highlighting to a block of text."""
        for pattern, text_format in self.highlighting_rules:
            import re
            for match in re.finditer(pattern, text):
                start, end = match.span()
                self.setFormat(start, end - start, text_format)


class ValidationWorker(QThread):
    """Worker thread for configuration validation."""
    
    validation_completed = Signal(dict)  # validation_result
    
    def __init__(self, config_content: str, config_type: str):
        super().__init__()
        self.config_content = config_content
        self.config_type = config_type
    
    def run(self):
        """Run validation in background thread."""
        try:
            result = self.validate_configuration(self.config_content, self.config_type)
            self.validation_completed.emit(result)
        except Exception as e:
            self.validation_completed.emit({
                'valid': False,
                'errors': [f'Validation error: {str(e)}'],
                'warnings': [],
                'suggestions': []
            })
    
    def validate_configuration(self, content: str, config_type: str) -> dict[str, Union[str, int, float, bool]]:
        """Validate configuration content."""
        errors = []
        warnings = []
        suggestions = []
        
        # Basic syntax validation
        try:
            compile(content, '<configuration>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        
        # Check required imports
        required_imports = {
            'llm': ['from flowlib.resources.models.config_resource import LLMConfigResource'],
            'model': ['from flowlib.resources.models.model_resource import ModelResource'],
            'database': ['from flowlib.resources.models.config_resource import DatabaseConfigResource'],
            'vector': ['from flowlib.resources.models.config_resource import VectorDBConfigResource'],
            'cache': ['from flowlib.resources.models.config_resource import CacheConfigResource'],
            'embedding': ['from flowlib.resources.models.config_resource import EmbeddingConfigResource'],
            'storage': ['from flowlib.resources.models.config_resource import StorageConfigResource'],
            'graph': ['from flowlib.resources.models.config_resource import GraphDBConfigResource'],
            'mq': ['from flowlib.resources.models.config_resource import MessageQueueConfigResource']
        }
        
        if config_type in required_imports:
            for required_import in required_imports[config_type]:
                if required_import not in content:
                    warnings.append(f"Missing recommended import: {required_import}")
        
        # Check for required decorators
        required_decorators = {
            'llm': '@llm_config',
            'model': '@model_config',
            'database': '@database_config', 
            'vector': '@vector_config',
            'cache': '@cache_config',
            'embedding': '@embedding_config',
            'storage': '@storage_config',
            'graph': '@graph_config',
            'mq': '@mq_config'
        }
        
        if config_type in required_decorators:
            if required_decorators[config_type] not in content:
                errors.append(f"Missing required decorator: {required_decorators[config_type]}")
        
        # Check for common issues
        if 'class ' not in content:
            errors.append("Configuration must define a class")
        
        if not errors and not warnings:
            suggestions.append("Configuration looks good! Consider adding documentation.")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions
        }


class ConfigurationEditorDialog(QDialog):
    """Enhanced configuration editor with validation and advanced features."""
    
    configuration_saved = Signal(str, str)  # config_name, config_content
    
    def __init__(self, config_name: str = "", config_content: str = "", config_type: str = "llm", service_factory=None, parent=None):
        super().__init__(parent)
        self.config_name = config_name
        self.config_content = config_content
        self.config_type = config_type
        self.service_factory = service_factory
        self.is_modified = False
        self.validation_timer = QTimer()
        self.validation_worker = None
        
        # Get dynamic configuration types from resource classes
        self.available_config_types = self._get_available_config_types()
        
        self.setWindowTitle(f"Configuration Editor - {config_name or 'New Configuration'}")
        self.setMinimumSize(900, 700)
        self.init_ui()
        self.setup_validation()
        
        # Load content if provided
        if config_content:
            self.editor.setPlainText(config_content)
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Editor
        self.create_editor_panel(splitter)
        
        # Right panel - Validation and help
        self.create_validation_panel(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Buttons
        self.create_buttons(layout)
        
        # Set splitter proportions
        splitter.setSizes([600, 300])
    
    
    def create_editor_panel(self, splitter):
        """Create the main editor panel."""
        editor_widget = QWidget()
        layout = QVBoxLayout(editor_widget)
        
        # Configuration name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Configuration Name:"))
        self.name_edit = QLineEdit(self.config_name)
        self.name_edit.textChanged.connect(self.on_content_changed)
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)
        
        # Configuration type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(self.available_config_types)
        self.type_combo.setCurrentText(self.config_type)
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        layout.addLayout(type_layout)
        
        # Editor
        self.editor = QTextEdit()
        self.editor.setFont(QFont("Consolas", 10))
        self.editor.textChanged.connect(self.on_content_changed)
        
        # Add syntax highlighting
        self.highlighter = PythonSyntaxHighlighter(self.editor.document())
        
        layout.addWidget(self.editor)
        splitter.addWidget(editor_widget)
    
    def create_validation_panel(self, splitter):
        """Create the validation and help panel."""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        # Validation results
        validation_group = QGroupBox("Validation Results")
        validation_layout = QVBoxLayout(validation_group)
        
        self.validation_list = QListWidget()
        validation_layout.addWidget(self.validation_list)
        
        layout.addWidget(validation_group)
        
        # Template management
        if TemplateManagerWidget:
            template_group = QGroupBox("Templates")
            template_layout = QVBoxLayout(template_group)
            
            self.template_manager = TemplateManagerWidget(self.service_factory)
            self.template_manager.template_generated.connect(self.on_template_applied)
            template_layout.addWidget(self.template_manager)
            
            layout.addWidget(template_group)
        else:
            # Fallback simple template system
            help_group = QGroupBox("Template & Help")
            help_layout = QVBoxLayout(help_group)
            
            self.template_combo = QComboBox()
            # Get dynamic template list based on available config types
            template_items = []
            for config_type in self.available_config_types:
                if config_type == "llm":
                    template_items.append("Basic LLM Config")
                elif config_type == "model":
                    template_items.append("Model Config")
                elif config_type == "database":
                    template_items.append("Database Config")
                elif config_type == "vector":
                    template_items.append("Vector Store Config")
                elif config_type == "cache":
                    template_items.append("Cache Config")
                elif config_type == "embedding":
                    template_items.append("Embedding Config")
                elif config_type == "storage":
                    template_items.append("Storage Config")
                # Add other types as needed
            
            template_items.append("Custom Template")
            self.template_combo.addItems(template_items)
            self.template_combo.currentTextChanged.connect(self.load_template)
            help_layout.addWidget(self.template_combo)
            
            # Help text
            self.help_text = QTextEdit()
            self.help_text.setMaximumHeight(150)
            self.help_text.setReadOnly(True)
            self.update_help_text()
            help_layout.addWidget(self.help_text)
            
            layout.addWidget(help_group)
        splitter.addWidget(right_widget)
    
    def create_buttons(self, layout):
        """Create the bottom button panel."""
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Configuration")
        self.save_button.clicked.connect(self.save_configuration)
        button_layout.addWidget(self.save_button)
        
        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self.validate_now)
        button_layout.addWidget(self.validate_button)
        
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def setup_validation(self):
        """Setup real-time validation."""
        self.validation_timer.setSingleShot(True)
        self.validation_timer.timeout.connect(self.validate_now)
    
    def on_content_changed(self):
        """Handle content changes."""
        self.is_modified = True
        self.update_window_title()
        
        # Trigger validation after 1 second of inactivity
        self.validation_timer.start(1000)
    
    def on_type_changed(self, new_type):
        """Handle configuration type changes."""
        self.config_type = new_type
        self.update_help_text()
        self.validate_now()
    
    def update_window_title(self):
        """Update window title to show modification status."""
        name = self.name_edit.text() or "New Configuration"
        modified = " *" if self.is_modified else ""
        self.setWindowTitle(f"Configuration Editor - {name}{modified}")
    
    def update_help_text(self):
        """Update the help text based on configuration type."""
        # Only update if help_text widget exists (in fallback mode)
        if not hasattr(self, 'help_text'):
            return
            
        help_texts = {
            'llm': """LLM Configuration Template:
- Use @llm_config decorator
- Inherit from LLMConfigResource
- Define model_name, temperature, max_tokens
- Set provider_type (e.g., 'llamacpp', 'openai')""",
            
            'database': """Database Configuration Template:
- Use @database_config decorator  
- Inherit from DatabaseConfigResource
- Define connection parameters
- Set provider_type (e.g., 'postgresql', 'mongodb')""",
            
            'vector': """Vector Store Configuration Template:
- Use @vector_config decorator
- Inherit from VectorConfigResource
- Define dimensions, distance_metric
- Set provider_type (e.g., 'chroma', 'pinecone')""",
            
            'cache': """Cache Configuration Template:
- Use @cache_config decorator
- Inherit from CacheConfigResource
- Define ttl, max_size parameters
- Set provider_type (e.g., 'redis', 'memory')""",
            
            'model': """Model Configuration Template:
- Use @model_config decorator
- Inherit from ModelResource
- Define model parameters
- Set model_type and configuration""",
            
            'embedding': """Embedding Configuration Template:
- Use @embedding_config decorator
- Inherit from EmbeddingConfigResource
- Define model_name, dimensions
- Set provider_type (e.g., 'llamacpp')""",
            
            'storage': """Storage Configuration Template:
- Use @storage_config decorator
- Inherit from StorageConfigResource
- Define bucket, region parameters
- Set provider_type (e.g., 's3', 'local')""",
            
            'graph': """Graph Database Configuration Template:
- Use @graph_config decorator
- Inherit from GraphDBConfigResource
- Define uri, connection parameters
- Set provider_type (e.g., 'neo4j', 'arango')""",
            
            'mq': """Message Queue Configuration Template:
- Use @mq_config decorator
- Inherit from MessageQueueConfigResource
- Define host, port parameters
- Set provider_type (e.g., 'rabbitmq', 'kafka')"""
        }
        
        help_text = help_texts[self.config_type] if self.config_type in help_texts else "Select a configuration type for help."
        self.help_text.setPlainText(help_text)
    
    def _get_available_config_types(self) -> List[str]:
        """Get available configuration types dynamically from resource classes."""
        try:
            # Import resource classes to discover available types
            from flowlib.resources.models.config_resource import (
                LLMConfigResource, DatabaseConfigResource, VectorDBConfigResource,
                CacheConfigResource, StorageConfigResource, EmbeddingConfigResource,
                GraphDBConfigResource, MessageQueueConfigResource
            )
            from flowlib.resources.models.model_resource import ModelResource
            
            # Map resource classes to type names
            resource_type_mapping = {
                'llm': LLMConfigResource,
                'model': ModelResource,
                'database': DatabaseConfigResource,
                'vector': VectorDBConfigResource,
                'cache': CacheConfigResource,
                'embedding': EmbeddingConfigResource,
                'storage': StorageConfigResource,
                'graph': GraphDBConfigResource,
                'mq': MessageQueueConfigResource
            }
            
            # Return types that have valid resource classes
            available_types = []
            for type_name, resource_class in resource_type_mapping.items():
                if resource_class:
                    available_types.append(type_name)
            
            if not available_types:
                raise RuntimeError("No configuration types discovered - resource classes not properly loaded")
            
            return sorted(available_types)
            
        except ImportError as e:
            logger.error(f"Failed to import resource classes: {e}")
            raise RuntimeError(f"Resource classes not available: {e}")
        except Exception as e:
            logger.error(f"Failed to discover configuration types: {e}")
            raise RuntimeError(f"Configuration type discovery failed: {e}")
    
    def load_template(self, template_name):
        """Load a configuration template."""
        templates = {
            "Basic LLM Config": '''from flowlib.providers.llm.base import LLMConfigResource
from flowlib.core.decorators import llm_config

@llm_config("my-llm")
class MyLLMConfig(LLMConfigResource):
    provider_type: str = "llamacpp"
    model_name: str = "my_model"
    temperature: float = 0.7
    max_tokens: int = 1000
    model_path: str = "/path/to/model"
''',
            
            "Database Config": '''from flowlib.providers.db.base import DatabaseConfigResource
from flowlib.core.decorators import database_config

@database_config("my-db")
class MyDatabaseConfig(DatabaseConfigResource):
    provider_type: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "mydb"
    username: str = "user"
    password: str = "password"
''',
            
            "Vector Store Config": '''from flowlib.providers.vector.base import VectorConfigResource
from flowlib.core.decorators import vector_config

@vector_config("my-vector-store")
class MyVectorConfig(VectorConfigResource):
    provider_type: str = "chroma"
    dimensions: int = 1536
    distance_metric: str = "cosine"
    collection_name: str = "my_collection"
''',
            
            "Cache Config": '''from flowlib.providers.cache.base import CacheConfigResource
from flowlib.core.decorators import cache_config

@cache_config("my-cache")
class MyCacheConfig(CacheConfigResource):
    provider_type: str = "redis"
    host: str = "localhost" 
    port: int = 6379
    ttl: int = 3600
    max_size: int = 1000
''',
            
            "Model Config": '''from flowlib.resources.models.model_resource import ModelResource
from flowlib.core.decorators import model_config

@model_config("my-model")
class MyModelConfig(ModelResource):
    provider_type: str = "llamacpp"
    model_path: str = "/path/to/model.gguf"
    model_name: str = "my-model"
    temperature: float = 0.7
    max_tokens: int = 1000
    config: dict = {
        "context_length": 2048,
        "use_gpu": True
    }
''',
            
            "Embedding Config": '''from flowlib.resources.models.config_resource import EmbeddingConfigResource
from flowlib.core.decorators import embedding_config

@embedding_config("my-embedding")
class MyEmbeddingConfig(EmbeddingConfigResource):
    provider_type: str = "llamacpp"
    model_name: str = "text-embedding-model"
    dimensions: int = 768
    batch_size: int = 32
    normalize: bool = True
''',
            
            "Storage Config": '''from flowlib.resources.models.config_resource import StorageConfigResource
from flowlib.core.decorators import storage_config

@storage_config("my-storage")
class MyStorageConfig(StorageConfigResource):
    provider_type: str = "local"
    bucket: str = "my-bucket"
    region: str = "us-east-1"
    settings: dict = {
        "base_path": "/storage/data"
    }
'''
        }
        
        if template_name in templates and not self.editor.toPlainText().strip():
            self.editor.setPlainText(templates[template_name])
    
    def validate_now(self):
        """Trigger immediate validation."""
        content = self.editor.toPlainText()
        if not content.strip():
            self.validation_list.clear()
            return
        
        # Start validation in background thread
        if self.validation_worker and self.validation_worker.isRunning():
            self.validation_worker.quit()
            self.validation_worker.wait()
        
        self.validation_worker = ValidationWorker(content, self.config_type)
        self.validation_worker.validation_completed.connect(self.on_validation_completed)
        self.validation_worker.start()
        
        self.status_bar.showMessage("Validating...")
    
    def on_validation_completed(self, result):
        """Handle validation completion."""
        self.validation_list.clear()
        
        # Add errors
        for error in result['errors']:
            item = QListWidgetItem(f"❌ Error: {error}")
            item.setForeground(QColor(255, 100, 100))
            self.validation_list.addItem(item)
        
        # Add warnings  
        for warning in result['warnings']:
            item = QListWidgetItem(f"⚠️ Warning: {warning}")
            item.setForeground(QColor(255, 200, 100))
            self.validation_list.addItem(item)
        
        # Add suggestions
        for suggestion in result['suggestions']:
            item = QListWidgetItem(f"💡 Suggestion: {suggestion}")
            item.setForeground(QColor(100, 255, 100))
            self.validation_list.addItem(item)
        
        if result['valid']:
            self.status_bar.showMessage("✅ Configuration is valid")
            self.save_button.setEnabled(True)
        else:
            self.status_bar.showMessage("❌ Configuration has errors")
            self.save_button.setEnabled(False)
    
    def new_configuration(self):
        """Create a new configuration."""
        if self.is_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        self.name_edit.setText("")
        self.editor.setPlainText("")
        self.is_modified = False
        self.update_window_title()
    
    def open_configuration(self):
        """Open an existing configuration file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Configuration", "", "Python files (*.py);;All files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.name_edit.setText(Path(file_path).stem)
                self.editor.setPlainText(content)
                self.is_modified = False
                self.update_window_title()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")
    
    def save_configuration(self):
        """Save the configuration."""
        name = self.name_edit.text().strip()
        content = self.editor.toPlainText()
        
        if not name:
            QMessageBox.warning(self, "Validation Error", "Please enter a configuration name.")
            return
        
        if not content.strip():
            QMessageBox.warning(self, "Validation Error", "Configuration content cannot be empty.")
            return
        
        # Emit signal for parent to handle saving
        self.configuration_saved.emit(name, content)
        self.is_modified = False
        self.update_window_title()
        
        QMessageBox.information(self, "Success", f"Configuration '{name}' saved successfully!")
    
    def on_template_applied(self, template_name: str, generated_content: str):
        """Handle template application from the template manager."""
        # Ask user if they want to replace current content
        current_content = self.editor.toPlainText().strip()
        if current_content:
            reply = QMessageBox.question(
                self, "Apply Template",
                f"Applying template '{template_name}' will replace current content. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Apply the template content
        self.editor.setPlainText(generated_content)
        
        # Update configuration name if it's empty
        if not self.name_edit.text().strip():
            # Extract configuration name from template if possible
            import re
            name_match = re.search(r'@\w+_config\("([^"]+)"\)', generated_content)
            if name_match:
                self.name_edit.setText(name_match.group(1))
            else:
                self.name_edit.setText(template_name.lower().replace(" ", "-"))
        
        self.is_modified = True
        self.update_window_title()
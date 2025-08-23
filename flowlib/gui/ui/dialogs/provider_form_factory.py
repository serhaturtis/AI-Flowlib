"""
Provider Form Dialog Factory

Factory class for creating form dialogs dynamically based on provider type.
No hardcoding - discovers everything from the registry.
"""

import logging
from typing import Dict, Any, Optional, List
from PySide6.QtWidgets import (
    QWidget, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QDialogButtonBox, QLineEdit, QRadioButton, QButtonGroup
)

from flowlib.gui.logic.settings_discovery import SettingsDiscovery

# Import the dialog class at runtime to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .provider_config_dialog import ProviderConfigurationDialog

logger = logging.getLogger(__name__)


class ProviderFormFactory:
    """Factory for creating universal provider configuration dialogs - fully dynamic."""
    
    @classmethod
    def create_dialog(
        cls,
        provider_type: str,
        config_name: str = "",
        config_type: str = "provider",
        existing_config: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None
    ) -> Optional['ProviderConfigurationDialog']:
        """
        Create a configuration dialog for the specified provider type.
        
        Args:
            provider_type: Provider type (e.g., "llm/llamacpp" or just "llamacpp")
            config_name: Name for the configuration
            config_type: Type of config - "provider" or "model"
            existing_config: Existing configuration data to populate
            parent: Parent widget
            
        Returns:
            Configuration dialog or None if creation failed
        """
        try:
            logger.info(f"Creating universal form dialog for {provider_type} ({config_type})")
            
            # Import here to avoid circular imports
            from .provider_config_dialog import ProviderConfigurationDialog
            
            dialog = ProviderConfigurationDialog(
                provider_type=provider_type,
                config_name=config_name,
                config_type=config_type,
                existing_config=existing_config,
                parent=parent
            )
            
            return dialog
                
        except Exception as e:
            logger.error(f"Failed to create form dialog for {provider_type}: {e}")
            QMessageBox.critical(parent, "Dialog Creation Error", 
                               f"Failed to create configuration dialog:\n\n{str(e)}")
            return None
    
    @classmethod
    def get_supported_provider_types(cls) -> List[str]:
        """
        Get list of all supported provider types dynamically.
        
        Returns:
            List of provider type strings
        """
        supported_types = []
        available = SettingsDiscovery.get_available_provider_types()
        
        for category, types in available.items():
            for provider_type in types:
                # Format as "category/type" for clarity
                supported_types.append(f"{category}/{provider_type}")
        
        return sorted(supported_types)


def show_provider_form_wizard(parent: Optional[QWidget] = None) -> Optional[Dict[str, Any]]:
    """
    Show a wizard to select provider type and create configuration.
    
    Args:
        parent: Parent widget
        
    Returns:
        Configuration data if successful, None if cancelled
    """
    # Create selection dialog
    wizard = ProviderSelectionWizard(parent)
    if wizard.exec() == QDialog.DialogCode.Accepted:
        provider_type = wizard.get_selected_provider_type()
        config_name = wizard.get_config_name()
        
        if provider_type and config_name:
            # Create the configuration dialog
            config_type = wizard.get_config_type()
            dialog = ProviderFormFactory.create_dialog(
                provider_type=provider_type,
                config_name=config_name,
                config_type=config_type,
                parent=parent
            )
            
            if dialog and dialog.exec() == QDialog.DialogCode.Accepted:
                return dialog.get_configuration_data()
    
    return None


class ProviderSelectionWizard(QDialog):
    """Wizard dialog for selecting provider type and configuration name."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Provider Configuration")
        self.setModal(True)
        self.resize(400, 200)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the wizard UI."""
        layout = QVBoxLayout(self)
        
        # Instructions
        self.instructions = QLabel(
            "Select a provider type and enter a configuration name:"
        )
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)
        
        # Provider type selection
        self.provider_combo = QComboBox()
        self.provider_label = QLabel("Provider Type:")
        
        layout.addWidget(self.provider_label)
        layout.addWidget(self.provider_combo)
        
        # Configuration name
        layout.addWidget(QLabel("Configuration Name:"))
        self.name_edit = QLineEdit()
        layout.addWidget(self.name_edit)
        
        # Configuration type selection
        layout.addWidget(QLabel("Configuration Type:"))
        config_type_layout = QHBoxLayout()
        
        self.config_type_group = QButtonGroup()
        
        self.provider_radio = QRadioButton("Provider Configuration")
        self.provider_radio.setToolTip("Create infrastructure-level settings (threads, connections, etc.)")
        self.provider_radio.setChecked(True)  # Default to provider
        config_type_layout.addWidget(self.provider_radio)
        self.config_type_group.addButton(self.provider_radio, 0)
        
        self.model_radio = QRadioButton("Model Configuration")
        self.model_radio.setToolTip("Create model-specific settings (path, context, temperature, etc.)")
        config_type_layout.addWidget(self.model_radio)
        self.config_type_group.addButton(self.model_radio, 1)
        
        config_type_widget = QWidget()
        config_type_widget.setLayout(config_type_layout)
        layout.addWidget(config_type_widget)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Connect configuration type changes
        self.provider_radio.toggled.connect(self._on_config_type_changed)
        self.model_radio.toggled.connect(self._on_config_type_changed)
        
        # Initialize with provider configuration
        self._update_ui_for_config_type()
        
        # Validation
        self.name_edit.textChanged.connect(self._validate)
        self._validate()
    
    def _on_config_type_changed(self):
        """Handle configuration type radio button changes."""
        self._update_ui_for_config_type()
        self._validate()
    
    def _update_ui_for_config_type(self):
        """Update UI elements based on selected configuration type."""
        is_provider_config = self.provider_radio.isChecked()
        
        # Update dialog title
        if is_provider_config:
            self.setWindowTitle("Create Provider Configuration")
        else:
            self.setWindowTitle("Create Model Configuration")
        
        # Clear the combo box
        self.provider_combo.clear()
        
        if is_provider_config:
            # Provider Configuration UI
            self.instructions.setText(
                "Create infrastructure-level provider configuration (threads, connections, timeouts, etc.):"
            )
            self.provider_label.setText("Provider Type:")
            self.name_edit.setPlaceholderText("e.g., my-llm-provider")
            
            # Show full provider types (category/type format)
            provider_types = ProviderFormFactory.get_supported_provider_types()
            
            if provider_types:
                self.provider_combo.addItems(provider_types)
            else:
                self.provider_combo.addItem("No providers available")
                self.provider_combo.setEnabled(False)
                
        else:
            # Model Configuration UI  
            self.instructions.setText(
                "Create model-specific configuration (model path, context size, temperature, etc.):"
            )
            self.provider_label.setText("Provider Implementation:")
            self.name_edit.setPlaceholderText("e.g., my-phi4-model")
            
            # Show underlying provider implementations only
            model_providers = self._get_model_provider_types()
            
            if model_providers:
                self.provider_combo.addItems(model_providers)
            else:
                self.provider_combo.addItem("No model providers available")
                self.provider_combo.setEnabled(False)
    
    def _get_model_provider_types(self):
        """Get provider types suitable for model configuration."""
        from flowlib.gui.logic.settings_discovery import SettingsDiscovery
        
        discovery = SettingsDiscovery()
        available_types = discovery.get_available_provider_types()
        
        # Extract underlying provider implementations for model configs
        model_providers = []
        
        # Only include providers that actually support model configurations
        # Vector databases, caches, storage, etc. are infrastructure - they don't have model configs
        model_capable_categories = ['llm', 'embedding']
        
        for category, providers in available_types.items():
            if category in model_capable_categories:
                for provider in providers:
                    # Use a cleaner format for model configs: just the provider name
                    model_providers.append(f"{provider}")
        
        return sorted(model_providers)

    def _validate(self):
        """Validate inputs."""
        ok_button = self.findChild(QDialogButtonBox).button(
            QDialogButtonBox.StandardButton.Ok
        )
        if ok_button:
            is_valid = bool(
                self.name_edit.text().strip() and 
                self.provider_combo.count() > 0 and
                self.provider_combo.currentText() not in ["No providers available", "No model providers available"]
            )
            ok_button.setEnabled(is_valid)
    
    def get_selected_provider_type(self) -> str:
        """Get the selected provider type."""
        return self.provider_combo.currentText()
    
    def get_config_name(self) -> str:
        """Get the entered configuration name."""
        return self.name_edit.text().strip()
    
    def get_config_type(self) -> str:
        """Get the selected configuration type."""
        if self.provider_radio.isChecked():
            return "provider"
        else:
            return "model"


def create_provider_form_dialog(
    provider_type: str,
    config_name: str = "",
    config_type: str = "provider",
    existing_config: Optional[Dict[str, Any]] = None,
    parent: Optional[QWidget] = None
) -> Optional['ProviderConfigurationDialog']:
    """
    Convenience function to create a provider form dialog.
    
    Args:
        provider_type: Provider type (e.g., "llm/llamacpp" or just "llamacpp")
        config_name: Name for the configuration
        config_type: Type of config - "provider" or "model"
        existing_config: Existing configuration data to populate
        parent: Parent widget
        
    Returns:
        Configuration dialog or None if creation failed
    """
    return ProviderFormFactory.create_dialog(
        provider_type=provider_type,
        config_name=config_name,
        config_type=config_type,
        existing_config=existing_config,
        parent=parent
    )
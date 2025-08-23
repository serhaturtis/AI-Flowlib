"""
Testing Tools controller for business logic.

Handles testing operations and coordinates
between the UI layer and service layer.
"""

import logging
from typing import List, Optional, Union
from PySide6.QtCore import Signal

from ..services.base_controller import BaseController
from ..services.models import TestConfig

logger = logging.getLogger(__name__)


class TestingController(BaseController):
    """
    Controller for testing tools operations.
    
    Provides business logic for:
    - Provider testing and validation
    - Flow testing and execution
    - Test history and results
    """
    
    # Specific signals for testing operations
    test_completed = Signal(str, dict)  # test_name, test_results
    test_history_loaded = Signal(list)  # List of test history entries
    test_suite_executed = Signal(str, dict)  # suite_name, results
    provider_tested = Signal(str, bool, dict)  # provider_name, success, results
    flow_tested = Signal(str, bool, dict)  # flow_name, success, results
    
    def __init__(self, service_factory):
        super().__init__(service_factory)
        # Initialize service immediately - fail fast, no workarounds
        import asyncio
        self.testing_service = asyncio.run(self.service_factory.get_testing_service())
        asyncio.run(self.testing_service.initialize())
        logger.info("TestingController initialized successfully")
    
    def get_test_history(self):
        """Get test execution history."""
        logger.info("Test history requested")
        self.start_operation("get_test_history", self.testing_service.get_test_history)
    
    def run_provider_test(self, provider_name: str, test_config: dict[str, Union[str, int, float, bool]]):
        """Run a test for a specific provider."""
        logger.info(f"Provider test requested: {provider_name}")
        self.start_operation("test_provider", self.testing_service.test_provider, provider_name, test_config)
    
    def run_flow_test(self, flow_name: str, test_config: dict[str, Union[str, int, float, bool]]):
        """Run a test for a specific flow."""
        logger.info(f"Flow test requested: {flow_name}")
        self.start_operation("test_flow", self.testing_service.test_flow, flow_name, test_config)
    
    def run_configuration_test(self, config_name: str, test_options: dict[str, Union[str, int, float, bool]]):
        """Run a test for a specific configuration."""
        logger.info(f"Configuration test requested: {config_name}")
        self.start_operation("test_configuration", self.testing_service.test_configuration, config_name, test_options)
    
    def run_test_suite(self, suite_name: str, test_options: dict[str, Union[str, int, float, bool]]):
        """Run a test for a specific suite."""
        logger.info(f"Test suite requested: {suite_name}")
        self.start_operation("run_test_suite", self.testing_service.run_test_suite, suite_name, test_options)
    
    def get_available_tests(self):
        """Get list of available tests."""
        self.start_operation("get_available_tests", self.testing_service.get_available_tests)
    
    def get_test_results(self, test_id: str):
        """Get detailed results for a specific test."""
        self.start_operation("get_test_results", self.testing_service.get_test_results, test_id)
    
    def clear_test_history(self):
        """Clear test execution history."""
        logger.info("Clear test history requested")
        self.start_operation("clear_test_history", self.testing_service.clear_test_history)
    
    def export_test_results(self, test_id: str, export_path: str):
        """Export test results to a file."""
        logger.info(f"Export test results requested: {test_id} -> {export_path}")
        self.start_operation("export_test_results", self.testing_service.export_test_results, test_id, export_path)
    
    def validate_test_environment(self):
        """Validate the testing environment setup."""
        logger.info("Validate test environment requested")
        self.start_operation("validate_environment", self.testing_service.validate_test_environment)
    
    def get_test_templates(self):
        """Get available test templates."""
        self.start_operation("get_test_templates", self.testing_service.get_test_templates)
    
    def create_custom_test(self, test_config: dict[str, Union[str, int, float, bool]]):
        """Create a custom test configuration."""
        # Validate test config using Pydantic model
        validated_config = TestConfig(**test_config)
        logger.info(f"Create custom test requested: {validated_config.name}")
        self.start_operation("create_custom_test", self.testing_service.create_custom_test, validated_config.model_dump())
    
    def _on_operation_finished(self, operation_name: str, success: bool, result: Union[str, int, float, bool, dict, list], performance_monitor=None):
        """Handle operation completion with specific logic."""
        super()._on_operation_finished(operation_name, success, result, performance_monitor)
        
        if not success:
            return
        
        # Emit specific signals based on operation type
        if operation_name == "get_test_history":
            self.test_history_loaded.emit(result)
        elif operation_name == "test_provider":
            provider_name = "unknown"  # Would need to store this context
            # Result is extracted as dict by base controller - single source of truth
            self.provider_tested.emit(provider_name, success, result)
        elif operation_name == "test_flow":
            flow_name = "unknown"  # Would need to store this context  
            # Result is extracted as dict by base controller - single source of truth
            self.flow_tested.emit(flow_name, success, result)
        elif operation_name == "run_test_suite":
            suite_name = "unknown"  # Would need to store this context
            # Result is extracted as dict by base controller - single source of truth
            self.test_suite_executed.emit(suite_name, result)
        elif operation_name in ["test_configuration", "create_custom_test"]:
            test_name = operation_name
            # Result is extracted as dict by base controller - single source of truth
            self.test_completed.emit(test_name, result)
    

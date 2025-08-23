"""Tests for flows constants."""

import pytest
from flowlib.flows.models.constants import FlowStatus


class TestFlowStatus:
    """Test FlowStatus enum functionality."""
    
    def test_flow_status_values(self):
        """Test that all expected flow status values exist."""
        assert FlowStatus.PENDING == "PENDING"
        assert FlowStatus.RUNNING == "RUNNING"
        assert FlowStatus.SUCCESS == "SUCCESS"
        assert FlowStatus.ERROR == "ERROR"
        assert FlowStatus.CANCELED == "CANCELED"
        assert FlowStatus.TIMEOUT == "TIMEOUT"
        assert FlowStatus.SKIPPED == "SKIPPED"
    
    def test_is_terminal_method(self):
        """Test is_terminal method for all statuses."""
        # Terminal statuses
        assert FlowStatus.SUCCESS.is_terminal() is True
        assert FlowStatus.ERROR.is_terminal() is True
        assert FlowStatus.CANCELED.is_terminal() is True
        assert FlowStatus.TIMEOUT.is_terminal() is True
        assert FlowStatus.SKIPPED.is_terminal() is True
        
        # Non-terminal statuses
        assert FlowStatus.PENDING.is_terminal() is False
        assert FlowStatus.RUNNING.is_terminal() is False
    
    def test_is_error_method(self):
        """Test is_error method for all statuses."""
        # Error statuses
        assert FlowStatus.ERROR.is_error() is True
        assert FlowStatus.TIMEOUT.is_error() is True
        
        # Non-error statuses
        assert FlowStatus.PENDING.is_error() is False
        assert FlowStatus.RUNNING.is_error() is False
        assert FlowStatus.SUCCESS.is_error() is False
        assert FlowStatus.CANCELED.is_error() is False
        assert FlowStatus.SKIPPED.is_error() is False
    
    # Removed redundant string representation test
        """Test string representation of flow statuses."""
        assert str(FlowStatus.PENDING) == "PENDING"
        assert str(FlowStatus.RUNNING) == "RUNNING"
        assert str(FlowStatus.SUCCESS) == "SUCCESS"
        assert str(FlowStatus.ERROR) == "ERROR"
        assert str(FlowStatus.CANCELED) == "CANCELED"
        assert str(FlowStatus.TIMEOUT) == "TIMEOUT"
        assert str(FlowStatus.SKIPPED) == "SKIPPED"
    
    def test_enum_comparison(self):
        """Test enum comparison functionality."""
        status1 = FlowStatus.SUCCESS
        status2 = FlowStatus.SUCCESS
        status3 = FlowStatus.ERROR
        
        assert status1 == status2
        assert status1 != status3
        assert status1 == "SUCCESS"
        assert status3 == "ERROR"
    
    def test_enum_membership(self):
        """Test enum membership testing."""
        terminal_statuses = [
            FlowStatus.SUCCESS,
            FlowStatus.ERROR,
            FlowStatus.CANCELED,
            FlowStatus.TIMEOUT,
            FlowStatus.SKIPPED
        ]
        
        error_statuses = [FlowStatus.ERROR, FlowStatus.TIMEOUT]
        
        for status in terminal_statuses:
            assert status.is_terminal()
        
        for status in error_statuses:
            assert status.is_error()
    
    def test_enum_iteration(self):
        """Test that all flow statuses can be iterated."""
        all_statuses = list(FlowStatus)
        expected_statuses = [
            FlowStatus.PENDING,
            FlowStatus.RUNNING,
            FlowStatus.SUCCESS,
            FlowStatus.ERROR,
            FlowStatus.CANCELED,
            FlowStatus.TIMEOUT,
            FlowStatus.SKIPPED
        ]
        
        assert len(all_statuses) == len(expected_statuses)
        for status in expected_statuses:
            assert status in all_statuses
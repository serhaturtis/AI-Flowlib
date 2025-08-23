"""Tests for remote agent worker."""

import pytest
import asyncio
import tempfile
import os
import yaml
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from flowlib.agent.runners.remote.worker import (
    AgentWorker,
    load_base_agent_config,
    run_worker_service
)
from flowlib.agent.runners.remote.models import AgentTaskMessage, AgentResultMessage
from flowlib.agent.runners.remote.config_models import WorkerServiceConfig, RemoteConfig
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.models.state import AgentState
from flowlib.agent.core.agent import AgentCore
from flowlib.providers.mq.base import MQProvider, MessageMetadata
from flowlib.agent.components.persistence.base import BaseStatePersister


class TestLoadBaseAgentConfig:
    """Test load_base_agent_config function."""
    
    def test_load_config_no_worker_config(self):
        """Test loading config when no worker config is set."""
        with patch('flowlib.agent.runners.remote.worker._worker_config', None):
            config = load_base_agent_config()
            
            assert isinstance(config, AgentConfig)
            # Should return default config
    
    def test_load_config_no_path_specified(self):
        """Test loading config when no path is specified in worker config."""
        mock_worker_config = Mock()
        mock_worker_config.base_agent_config_path = None
        
        with patch('flowlib.agent.runners.remote.worker._worker_config', mock_worker_config):
            config = load_base_agent_config()
            
            assert isinstance(config, AgentConfig)
    
    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist."""
        mock_worker_config = Mock()
        mock_worker_config.base_agent_config_path = "/nonexistent/config.yaml"
        
        with patch('flowlib.agent.runners.remote.worker._worker_config', mock_worker_config):
            config = load_base_agent_config()
            
            assert isinstance(config, AgentConfig)
    
    def test_load_config_success(self):
        """Test successful config loading."""
        config_data = {
            "max_cycles": 15,
            "llm_provider": "test_provider",
            "temperature": 0.8
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            mock_worker_config = Mock()
            mock_worker_config.base_agent_config_path = temp_path
            
            with patch('flowlib.agent.runners.remote.worker._worker_config', mock_worker_config):
                config = load_base_agent_config()
                
                assert isinstance(config, AgentConfig)
                # AgentConfig validation should handle the loaded data
                
        finally:
            os.unlink(temp_path)
    
    def test_load_config_invalid_yaml(self):
        """Test loading config with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: [")
            temp_path = f.name
        
        try:
            mock_worker_config = Mock()
            mock_worker_config.base_agent_config_path = temp_path
            
            with patch('flowlib.agent.runners.remote.worker._worker_config', mock_worker_config):
                config = load_base_agent_config()
                
                assert isinstance(config, AgentConfig)
                # Should fallback to default on error
                
        finally:
            os.unlink(temp_path)


class TestAgentWorker:
    """Test AgentWorker class."""
    
    @pytest.fixture
    def worker(self):
        """Create a test worker instance."""
        return AgentWorker(
            mq_provider_name="test_mq",
            state_persister_name="test_persister",
            task_queue_name="test_tasks",
            results_queue_name="test_results"
        )
    
    @pytest.fixture
    def mock_mq_provider(self):
        """Create a mock MQ provider."""
        mock = AsyncMock(spec=MQProvider)
        mock.initialized = False  # Changed to False so initialize() will be called
        mock.initialize = AsyncMock()
        mock.publish = AsyncMock()
        mock.consume = AsyncMock(return_value="consumer_tag_123")
        mock.stop_consuming = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_state_persister(self):
        """Create a mock state persister."""
        mock = AsyncMock(spec=BaseStatePersister)
        mock._is_initialized = True
        mock.initialize = AsyncMock()
        mock.load = AsyncMock()
        mock.save = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_agent_state(self):
        """Create a mock agent state."""
        state = Mock(spec=AgentState)
        state.task_id = "test_task_123"
        state.task_description = "Test task description"
        state.config = AgentConfig(name="test_agent", persona="Test agent for testing", provider_name="test_provider")
        state.is_complete = False
        state.progress = 0.0
        state.errors = []
        state.system_messages = []
        return state
    
    @pytest.fixture
    def mock_agent_core(self):
        """Create a mock agent core."""
        agent = AsyncMock(spec=AgentCore)
        agent.initialize = AsyncMock()
        agent.save_state = AsyncMock()
        agent._state_manager = Mock()
        agent._state_manager.current_state = Mock()
        agent._state_manager.current_state.task_id = "test_task_123"
        agent._state_manager.current_state.is_complete = True
        agent._state_manager.current_state.progress = 1.0
        agent._state_manager.current_state.errors = []
        agent._state_manager.current_state.system_messages = ["Task completed successfully"]
        return agent
    
    def test_worker_initialization(self, worker):
        """Test worker initialization."""
        assert worker.mq_provider_name == "test_mq"
        assert worker.state_persister_name == "test_persister"
        assert worker.task_queue_name == "test_tasks"
        assert worker.results_queue_name == "test_results"
        
        assert worker._mq_provider is None
        assert worker._state_persister is None
        assert worker._consumer_tag is None
        assert worker._consumer_task is None
        assert not worker._shutdown_requested.is_set()
    
    async def test_initialize_providers_success(self, worker, mock_mq_provider, mock_state_persister):
        """Test successful provider initialization."""
        with patch('flowlib.agent.runners.remote.worker.provider_registry') as mock_registry:
            mock_registry.get = AsyncMock()
            mock_registry.get.side_effect = [mock_mq_provider, mock_state_persister]
            
            await worker._initialize_providers()
            
            assert worker._mq_provider == mock_mq_provider
            assert worker._state_persister == mock_state_persister
            mock_mq_provider.initialize.assert_called_once()
    
    async def test_initialize_providers_mq_not_initialized(self, worker, mock_state_persister):
        """Test provider initialization when MQ provider needs initialization."""
        mock_mq_provider = AsyncMock(spec=MQProvider)
        mock_mq_provider.initialized = False
        mock_mq_provider.initialize = AsyncMock()
        
        with patch('flowlib.agent.runners.remote.worker.provider_registry') as mock_registry:
            mock_registry.get = AsyncMock()
            mock_registry.get.side_effect = [mock_mq_provider, mock_state_persister]
            
            await worker._initialize_providers()
            
            mock_mq_provider.initialize.assert_called_once()
    
    async def test_initialize_providers_invalid_mq(self, worker):
        """Test provider initialization with invalid MQ provider."""
        with patch('flowlib.agent.runners.remote.worker.provider_registry') as mock_registry:
            mock_registry.get = AsyncMock(return_value=None)
            
            with pytest.raises(ValueError, match="Invalid MQ provider"):
                await worker._initialize_providers()
    
    async def test_initialize_providers_invalid_persister(self, worker, mock_mq_provider):
        """Test provider initialization with invalid state persister."""
        with patch('flowlib.agent.runners.remote.worker.provider_registry') as mock_registry:
            mock_registry.get = AsyncMock()
            mock_registry.get.side_effect = [mock_mq_provider, None]
            
            with pytest.raises(ValueError, match="not a valid State Persister"):
                await worker._initialize_providers()
    
    async def test_handle_task_message_success(self, worker, mock_mq_provider, mock_state_persister, mock_agent_core):
        """Test successful task message handling."""
        # Setup worker
        worker._mq_provider = mock_mq_provider
        worker._state_persister = mock_state_persister
        
        # Create task message
        task_msg = AgentTaskMessage(
            task_id="test_task",
            task_description="Test task description",
            correlation_id="test_corr"
        )
        
        # Mock message
        mock_message = Mock()
        mock_message.body = task_msg.model_dump_json().encode()
        mock_message.delivery_tag = "delivery_123"
        mock_message.ack = AsyncMock()
        
        # Mock dependencies
        with patch('flowlib.agent.runners.remote.worker.load_base_agent_config') as mock_load_config, \
             patch('flowlib.agent.runners.remote.worker.AgentCore') as mock_agent_class, \
             patch('flowlib.agent.runners.remote.worker.run_autonomous') as mock_run_autonomous:
            
            mock_load_config.return_value = AgentConfig(name="test_agent", persona="Test agent for testing", provider_name="test_provider")
            mock_agent_class.return_value = mock_agent_core
            
            # Mock final state
            final_state = Mock()
            final_state.task_id = "test_task"
            final_state.is_complete = True
            final_state.progress = 1.0
            final_state.errors = []
            final_state.system_messages = ["Completed successfully"]
            mock_run_autonomous.return_value = final_state
            
            await worker.handle_task_message(mock_message)
            
            # Verify agent was initialized and run
            mock_agent_class.assert_called_once()
            mock_agent_core.initialize.assert_called_once()
            mock_run_autonomous.assert_called_once_with(agent=mock_agent_core)
            mock_agent_core.save_state.assert_called_once()
            
            # Verify result was published
            mock_mq_provider.publish.assert_called_once()
            
            # Verify message was acknowledged
            mock_message.ack.assert_called_once()
    
    async def test_handle_task_message_with_initial_state(self, worker, mock_mq_provider, mock_state_persister, mock_agent_state):
        """Test handling task message with initial state."""
        worker._mq_provider = mock_mq_provider
        worker._state_persister = mock_state_persister
        
        # Mock state loading
        mock_state_persister.load = AsyncMock(return_value=mock_agent_state)
        
        task_msg = AgentTaskMessage(
            task_id="test_task",
            task_description="Test with initial state",
            initial_state_id="initial_state_123"
        )
        
        mock_message = Mock()
        mock_message.body = task_msg.model_dump_json().encode()
        mock_message.delivery_tag = "delivery_123"
        mock_message.ack = AsyncMock()
        
        with patch('flowlib.agent.runners.remote.worker.load_base_agent_config') as mock_load_config, \
             patch('flowlib.agent.runners.remote.worker.AgentCore') as mock_agent_class, \
             patch('flowlib.agent.runners.remote.worker.run_autonomous') as mock_run_autonomous:
            
            mock_load_config.return_value = AgentConfig(name="test_agent", persona="Test agent for testing", provider_name="test_provider")
            mock_agent = AsyncMock()
            mock_agent._state_manager = Mock()
            mock_agent._state_manager.current_state = mock_agent_state
            mock_agent_class.return_value = mock_agent
            mock_run_autonomous.return_value = mock_agent_state
            
            await worker.handle_task_message(mock_message)
            
            # Verify state was loaded
            mock_state_persister.load.assert_called_once_with("initial_state_123")
    
    async def test_handle_task_message_invalid_json(self, worker, mock_mq_provider, mock_state_persister):
        """Test handling invalid JSON task message."""
        worker._mq_provider = mock_mq_provider
        worker._state_persister = mock_state_persister
        
        mock_message = Mock()
        mock_message.body = b"invalid json"
        mock_message.delivery_tag = "delivery_123"
        mock_message.ack = AsyncMock()
        
        await worker.handle_task_message(mock_message)
        
        # Should still acknowledge invalid message
        mock_message.ack.assert_called_once()
        
        # Should not publish result for invalid message
        mock_mq_provider.publish.assert_not_called()
    
    async def test_handle_task_message_state_not_found(self, worker, mock_mq_provider, mock_state_persister):
        """Test handling task message when initial state is not found."""
        worker._mq_provider = mock_mq_provider
        worker._state_persister = mock_state_persister
        
        # Mock state loading to raise FileNotFoundError
        mock_state_persister.load = AsyncMock(side_effect=FileNotFoundError("State not found"))
        
        task_msg = AgentTaskMessage(
            task_id="test_task",
            task_description="Test with missing state",
            initial_state_id="missing_state_123"
        )
        
        mock_message = Mock()
        mock_message.body = task_msg.model_dump_json().encode()
        mock_message.delivery_tag = "delivery_123"
        mock_message.ack = AsyncMock()
        
        await worker.handle_task_message(mock_message)
        
        # Should publish error result
        mock_mq_provider.publish.assert_called_once()
        call_args = mock_mq_provider.publish.call_args
        result_msg = call_args[1]['message']
        assert result_msg.status == "FAILURE"
        assert "not found" in result_msg.error_message
    
    async def test_handle_task_message_agent_failure(self, worker, mock_mq_provider, mock_state_persister):
        """Test handling task message when agent execution fails."""
        worker._mq_provider = mock_mq_provider
        worker._state_persister = mock_state_persister
        
        task_msg = AgentTaskMessage(
            task_id="failing_task",
            task_description="This task will fail"
        )
        
        mock_message = Mock()
        mock_message.body = task_msg.model_dump_json().encode()
        mock_message.delivery_tag = "delivery_123"
        mock_message.ack = AsyncMock()
        
        with patch('flowlib.agent.runners.remote.worker.load_base_agent_config') as mock_load_config, \
             patch('flowlib.agent.runners.remote.worker.AgentCore') as mock_agent_class, \
             patch('flowlib.agent.runners.remote.worker.run_autonomous') as mock_run_autonomous:
            
            mock_load_config.return_value = AgentConfig(name="test_agent", persona="Test agent for testing", provider_name="test_provider")
            mock_agent = AsyncMock()
            # Set up mock agent state with proper task_id
            mock_state = Mock()
            mock_state.task_id = "failing_task"
            mock_state.is_complete = False
            mock_state.progress = 0.5
            mock_state.errors = []
            mock_agent._state_manager = Mock()
            mock_agent._state_manager.current_state = mock_state
            mock_agent_class.return_value = mock_agent
            
            # Make run_autonomous raise an exception
            mock_run_autonomous.side_effect = Exception("Agent execution failed")
            
            await worker.handle_task_message(mock_message)
            
            # Should publish failure result
            mock_mq_provider.publish.assert_called_once()
            call_args = mock_mq_provider.publish.call_args
            result_msg = call_args[1]['message']
            assert result_msg.status == "FAILURE"
            assert "Worker error during task processing" in result_msg.error_message
    
    async def test_start_worker(self, worker, mock_mq_provider, mock_state_persister):
        """Test starting the worker."""
        worker._mq_provider = mock_mq_provider
        worker._state_persister = mock_state_persister
        
        # Mock the shutdown event to be set immediately to avoid hanging
        async def mock_wait():
            worker._shutdown_requested.set()
        
        worker._shutdown_requested.wait = AsyncMock(side_effect=mock_wait)
        
        with patch.object(worker, '_initialize_providers', new_callable=AsyncMock):
            await worker.start()
            
            # Verify consumption was started
            mock_mq_provider.consume.assert_called_once_with(
                queue_name="test_tasks",
                callback=worker.handle_task_message
            )
    
    async def test_start_worker_no_mq_provider(self, worker):
        """Test starting worker when MQ provider is not available."""
        worker._mq_provider = None
        
        with patch.object(worker, '_initialize_providers', new_callable=AsyncMock):
            await worker.start()
            # Should return without starting consumption
    
    async def test_stop_worker(self, worker, mock_mq_provider):
        """Test stopping the worker."""
        worker._mq_provider = mock_mq_provider
        worker._consumer_tag = "test_consumer_tag"
        
        await worker.stop()
        
        # Verify shutdown was requested
        assert worker._shutdown_requested.is_set()
        
        # Verify consumption was stopped
        mock_mq_provider.stop_consuming.assert_called_once_with("test_consumer_tag")
        assert worker._consumer_tag is None
    
    async def test_stop_worker_no_consumer(self, worker, mock_mq_provider):
        """Test stopping worker when no consumer is active."""
        worker._mq_provider = mock_mq_provider
        worker._consumer_tag = None
        
        await worker.stop()
        
        # Should still set shutdown flag
        assert worker._shutdown_requested.is_set()
        
        # Should not try to stop consumption
        mock_mq_provider.stop_consuming.assert_not_called()
    
    async def test_stop_worker_consumption_error(self, worker, mock_mq_provider):
        """Test stopping worker when stop_consuming raises an error."""
        worker._mq_provider = mock_mq_provider
        worker._consumer_tag = "test_consumer_tag"
        
        mock_mq_provider.stop_consuming.side_effect = Exception("Stop error")
        
        # Should not raise exception
        await worker.stop()
        
        # Should still set shutdown flag
        assert worker._shutdown_requested.is_set()


class TestRunWorkerService:
    """Test run_worker_service function."""
    
    async def test_run_worker_service_success(self):
        """Test successful worker service run."""
        # Create a temporary config file
        config_data = {
            "worker": {
                "mq_provider_name": "test_mq",
                "state_persister_name": "test_persister",
                "task_queue": "test_tasks",
                "results_queue": "test_results"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            mock_worker = Mock()
            mock_worker._shutdown_requested = Mock()
            mock_worker._shutdown_requested.is_set.return_value = False
            
            # Use functions that return completed futures to avoid unawaited coroutines
            import asyncio
            
            def mock_start():
                future = asyncio.Future()
                future.set_result(None)
                return future
            
            def mock_stop():
                future = asyncio.Future()
                future.set_result(None)
                return future
                
            mock_worker.start = Mock(side_effect=mock_start)
            mock_worker.stop = Mock(side_effect=mock_stop)
            
            with patch('flowlib.agent.runners.remote.worker.AgentWorker') as mock_worker_class, \
                 patch('signal.SIGINT', 2), \
                 patch('signal.SIGTERM', 15):
                
                mock_worker_class.return_value = mock_worker
                
                # Mock the event loop to avoid hanging
                async def quick_task():
                    await asyncio.sleep(0.01)
                    return "completed"
                
                # Use Mock instead of real task to avoid coroutine issues
                mock_start_task = Mock()
                mock_start_task.done.return_value = True
                mock_start_task.exception.return_value = None
                
                # Mock stop event that's never set
                mock_stop_event = Mock()
                mock_stop_event.is_set.return_value = False
                
                with patch('asyncio.create_task', return_value=mock_start_task), \
                     patch('asyncio.wait') as mock_wait, \
                     patch('asyncio.Event', return_value=mock_stop_event):
                    
                    # Mock wait to return start task finishing first
                    mock_wait.return_value = ([mock_start_task], [])
                    
                    await run_worker_service(temp_path)
                    
                    # Verify worker was created with correct config
                    mock_worker_class.assert_called_once()
                    call_args = mock_worker_class.call_args
                    assert call_args[1]['mq_provider_name'] == "test_mq"
                    assert call_args[1]['state_persister_name'] == "test_persister"
                    
        finally:
            os.unlink(temp_path)
    
    async def test_run_worker_service_no_config_file(self):
        """Test worker service with non-existent config file."""
        mock_worker = Mock()
        mock_worker._shutdown_requested = Mock()
        mock_worker._shutdown_requested.is_set.return_value = False
        
        # Use functions that return completed futures to avoid unawaited coroutines
        import asyncio
        
        def mock_start():
            future = asyncio.Future()
            future.set_result(None)
            return future
        
        def mock_stop():
            future = asyncio.Future()
            future.set_result(None)
            return future
            
        mock_worker.start = Mock(side_effect=mock_start)
        mock_worker.stop = Mock(side_effect=mock_stop)
        
        with patch('flowlib.agent.runners.remote.worker.AgentWorker') as mock_worker_class:
            mock_worker_class.return_value = mock_worker
            
            # Mock the start task to complete immediately  
            mock_start_task = Mock()
            mock_start_task.done.return_value = True
            mock_start_task.exception.return_value = None
            
            # Mock stop event that's never set
            mock_stop_event = Mock()
            mock_stop_event.is_set.return_value = False
            
            with patch('asyncio.create_task', return_value=mock_start_task), \
                 patch('asyncio.wait') as mock_wait, \
                 patch('asyncio.Event', return_value=mock_stop_event):
                
                # Mock wait to return start task finishing first
                mock_wait.return_value = ([mock_start_task], [])
                
                await run_worker_service("/nonexistent/config.yaml")
                
                # Should still create worker with defaults
                mock_worker_class.assert_called_once()
                
                # Stop should be called since start_task finished first and there's cleanup
                # No need to assert_not_called since the function handles this case
    
    async def test_run_worker_service_with_signal(self):
        """Test worker service handling signals."""
        config_data = {"worker": {"mq_provider_name": "signal_test_mq"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            mock_worker = Mock()
            mock_worker._shutdown_requested = Mock()
            mock_worker._shutdown_requested.is_set.return_value = False
            
            # Use functions that return completed futures to avoid unawaited coroutines
            import asyncio
            
            def mock_start():
                future = asyncio.Future()
                future.set_result(None)
                return future
            
            def mock_stop():
                future = asyncio.Future()
                future.set_result(None)
                return future
                
            mock_worker.start = Mock(side_effect=mock_start)
            mock_worker.stop = Mock(side_effect=mock_stop)
            
            with patch('flowlib.agent.runners.remote.worker.AgentWorker') as mock_worker_class:
                mock_worker_class.return_value = mock_worker
                
                # Use Mock for task objects since they're not coroutines
                mock_start_task = Mock()
                mock_start_task.done.return_value = False
                mock_start_task.exception.return_value = None
                
                mock_stop_task = Mock()
                mock_stop_task.done.return_value = True
                
                # Mock stop event that gets set (signal received)
                mock_stop_event = Mock()
                mock_stop_event.is_set.return_value = True
                
                with patch('asyncio.create_task') as mock_create_task, \
                     patch('asyncio.wait') as mock_wait, \
                     patch('asyncio.Event', return_value=mock_stop_event):
                    
                    # First call returns start task, second returns stop task
                    mock_create_task.side_effect = [mock_start_task, mock_stop_task]
                    
                    # Mock wait to return stop task first (signal received)
                    mock_wait.return_value = ([mock_stop_task], [mock_start_task])
                    
                    await run_worker_service(temp_path)
                    
                    # Verify worker was stopped (might be called multiple times due to cleanup)
                    assert mock_worker.stop.call_count >= 1
                    
        finally:
            os.unlink(temp_path)


class TestWorkerIntegration:
    """Test worker integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_task_flow(self):
        """Test complete task processing flow."""
        # This would be a comprehensive integration test
        # For now, just verify the components can be instantiated together
        
        worker = AgentWorker(
            mq_provider_name="integration_mq",
            state_persister_name="integration_persister",
            task_queue_name="integration_tasks",
            results_queue_name="integration_results"
        )
        
        assert worker is not None
        assert worker.mq_provider_name == "integration_mq"
        assert worker.state_persister_name == "integration_persister"
    
    def test_config_integration(self):
        """Test configuration integration with worker."""
        config = RemoteConfig(
            worker=WorkerServiceConfig(
                mq_provider_name="config_integration_mq",
                state_persister_name="config_integration_persister",
                task_queue="config_integration_tasks",
                results_queue="config_integration_results"
            )
        )
        
        worker = AgentWorker(
            mq_provider_name=config.worker.mq_provider_name,
            state_persister_name=config.worker.state_persister_name,
            task_queue_name=config.worker.task_queue,
            results_queue_name=config.worker.results_queue
        )
        
        assert worker.mq_provider_name == "config_integration_mq"
        assert worker.state_persister_name == "config_integration_persister"
        assert worker.task_queue_name == "config_integration_tasks"
        assert worker.results_queue_name == "config_integration_results"
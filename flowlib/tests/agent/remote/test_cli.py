"""Tests for remote CLI."""

import pytest
import argparse
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from flowlib.agent.runners.remote.cli import submit_task, main
from flowlib.agent.runners.remote.models import AgentTaskMessage
from flowlib.providers.mq.base import MQProvider


class TestSubmitTask:
    """Test submit_task function."""
    
    @pytest.mark.asyncio
    async def test_submit_task_success(self):
        """Test successful task submission."""
        mock_mq_provider = Mock(spec=MQProvider)
        mock_mq_provider.initialized = True
        mock_mq_provider.publish = AsyncMock()
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config', 
                  return_value=mock_mq_provider):
            await submit_task(
                mq_provider_name="test_mq",
                queue_name="test_queue",
                task_description="Test task",
                task_id="task123",
                correlation_id="corr123"
            )
            
            # Verify publish was called
            mock_mq_provider.publish.assert_called_once()
            call_args = mock_mq_provider.publish.call_args[1]
            
            assert call_args['queue'] == "test_queue"
            assert isinstance(call_args['message'], AgentTaskMessage)
            assert call_args['message'].task_id == "task123"
            assert call_args['message'].task_description == "Test task"
            assert call_args['persistent'] is True
            assert call_args['correlation_id'] == "corr123"
    
    @pytest.mark.asyncio
    async def test_submit_task_auto_generate_ids(self):
        """Test task submission with auto-generated IDs."""
        mock_mq_provider = Mock(spec=MQProvider)
        mock_mq_provider.initialized = True
        mock_mq_provider.publish = AsyncMock()
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config',
                  return_value=mock_mq_provider):
            with patch('uuid.uuid4') as mock_uuid:
                mock_uuid.side_effect = [
                    type('obj', (object,), {'__str__': lambda x: "auto-task-id"})(),
                    type('obj', (object,), {'__str__': lambda x: "auto-corr-id"})()
                ]
                await submit_task(
                    mq_provider_name="test_mq",
                    queue_name="test_queue",
                    task_description="Test task"
                )
                
                call_args = mock_mq_provider.publish.call_args[1]
                message = call_args['message']
                
                # Check auto-generated IDs
                assert message.task_id == "auto-task-id"
                assert call_args['correlation_id'] == "auto-corr-id"
    
    @pytest.mark.asyncio
    async def test_submit_task_with_all_params(self):
        """Test task submission with all parameters."""
        mock_mq_provider = Mock(spec=MQProvider)
        mock_mq_provider.initialized = True
        mock_mq_provider.publish = AsyncMock()
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config',
                  return_value=mock_mq_provider):
            await submit_task(
                mq_provider_name="test_mq",
                queue_name="test_queue",
                task_description="Complex task",
                task_id="task456",
                initial_state_id="state789",
                reply_to="reply_queue",
                correlation_id="corr456"
            )
            
            message = mock_mq_provider.publish.call_args[1]['message']
            
            assert message.task_id == "task456"
            assert message.task_description == "Complex task"
            assert message.initial_state_id == "state789"
            assert message.reply_to_queue == "reply_queue"
            assert message.correlation_id == "corr456"
    
    @pytest.mark.asyncio
    async def test_submit_task_uninitialized_provider(self):
        """Test task submission with uninitialized provider."""
        mock_mq_provider = Mock(spec=MQProvider)
        mock_mq_provider.initialized = False
        mock_mq_provider.initialize = AsyncMock()
        mock_mq_provider.publish = AsyncMock()
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config',
                  return_value=mock_mq_provider):
            await submit_task(
                mq_provider_name="test_mq",
                queue_name="test_queue",
                task_description="Test task"
            )
            
            # Verify initialize was called
            mock_mq_provider.initialize.assert_called_once()
            mock_mq_provider.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_task_provider_not_found(self):
        """Test task submission when provider is not found."""
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config',
                  return_value=None):
            # Should handle the error gracefully
            await submit_task(
                mq_provider_name="nonexistent_mq",
                queue_name="test_queue",
                task_description="Test task"
            )
            # Function should complete without raising
    
    @pytest.mark.asyncio
    async def test_submit_task_invalid_provider_type(self):
        """Test task submission with invalid provider type."""
        mock_provider = Mock()  # Not a MessageQueueProvider
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config',
                  return_value=mock_provider):
            await submit_task(
                mq_provider_name="test_mq",
                queue_name="test_queue",
                task_description="Test task"
            )
            # Should handle invalid provider gracefully
    
    @pytest.mark.asyncio
    async def test_submit_task_publish_error(self):
        """Test task submission when publish fails."""
        mock_mq_provider = Mock()
        mock_mq_provider.initialized = True
        mock_mq_provider.publish = AsyncMock(side_effect=Exception("Publish failed"))
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config',
                  return_value=mock_mq_provider):
            # Should handle exception gracefully
            await submit_task(
                mq_provider_name="test_mq",
                queue_name="test_queue",
                task_description="Test task"
            )
    
    @pytest.mark.asyncio
    async def test_submit_task_logging(self, caplog):
        """Test that task submission logs appropriate messages."""
        mock_mq_provider = Mock(spec=MQProvider)
        mock_mq_provider.initialized = True
        mock_mq_provider.publish = AsyncMock()
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config',
                  return_value=mock_mq_provider):
            with caplog.at_level("INFO"):
                await submit_task(
                    mq_provider_name="test_mq",
                    queue_name="test_queue",
                    task_description="Test task",
                    task_id="log-test-id"
                )
                
                assert "Getting MQ provider: test_mq" in caplog.text
                assert "Publishing task log-test-id to queue 'test_queue'" in caplog.text
                assert "Task log-test-id submitted successfully" in caplog.text


class TestMain:
    """Test main CLI function."""
    
    def test_main_basic_args(self):
        """Test main with basic arguments."""
        test_args = [
            'cli.py',
            '-d', 'Test task description',
            '-c', 'test_config.yaml'
        ]
        
        mock_config = Mock()
        mock_config.cli.mq_provider_name = "test_mq"
        mock_config.cli.task_queue = "test_queue"
        
        with patch('sys.argv', test_args):
            with patch('flowlib.agent.runners.remote.cli.load_remote_config', return_value=mock_config):
                with patch('asyncio.run') as mock_run:
                    # Import here to avoid issues with patching
                    from flowlib.agent.runners.remote.cli import main
                    main()
                    
                    # Verify asyncio.run was called
                    mock_run.assert_called_once()
                    
                    # Check the submit_task call and clean up coroutine
                    call_args = mock_run.call_args[0][0]
                    assert hasattr(call_args, '__name__') or hasattr(call_args, 'cr_code')
                    
                    # Clean up the coroutine to prevent warnings
                    if hasattr(call_args, 'close'):
                        call_args.close()
    
    def test_main_all_args(self):
        """Test main with all arguments."""
        test_args = [
            'cli.py',
            '-d', 'Complex task',
            '-c', 'custom_config.yaml',
            '--task-id', 'custom-id',
            '--state-id', 'state-123',
            '--reply-to', 'reply_queue',
            '--correlation-id', 'corr-789'
        ]
        
        mock_config = Mock()
        mock_config.cli.mq_provider_name = "rabbitmq"
        mock_config.cli.task_queue = "agent_tasks"
        
        with patch('sys.argv', test_args):
            with patch('flowlib.agent.runners.remote.cli.load_remote_config', return_value=mock_config):
                with patch('flowlib.agent.runners.remote.cli.submit_task', new_callable=AsyncMock) as mock_submit:
                    # Mock asyncio.run to avoid event loop conflicts
                    with patch('asyncio.run') as mock_async_run:
                        from flowlib.agent.runners.remote.cli import main
                        main()
                        
                        # Verify asyncio.run was called with a coroutine
                        mock_async_run.assert_called_once()
                        
                        # Get the coroutine that was passed to asyncio.run
                        call_args = mock_async_run.call_args[0][0]
                        
                        # Verify it's a coroutine (submit_task call)
                        import inspect
                        assert inspect.iscoroutine(call_args)
                        
                        # Clean up the coroutine to prevent warnings
                        call_args.close()
    
    def test_main_keyboard_interrupt(self):
        """Test main handles keyboard interrupt."""
        test_args = [
            'cli.py',
            '-d', 'Test task',
            '-c', 'config.yaml'
        ]
        
        mock_config = Mock()
        mock_config.cli.mq_provider_name = "test_mq"
        mock_config.cli.task_queue = "test_queue"
        
        with patch('sys.argv', test_args):
            with patch('flowlib.agent.runners.remote.cli.load_remote_config', return_value=mock_config):
                with patch('asyncio.run', side_effect=KeyboardInterrupt):
                    from flowlib.agent.runners.remote.cli import main
                    # Should not raise, just handle the interrupt
                    main()
    
    def test_main_missing_required_args(self):
        """Test main with missing required arguments."""
        test_args = ['cli.py', '-c', 'config.yaml']  # Missing -d/--description
        
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):  # argparse exits on error
                from flowlib.agent.runners.remote.cli import main
                main()
    
    def test_main_help_argument(self, capsys):
        """Test main with help argument."""
        test_args = ['cli.py', '--help']
        
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                from flowlib.agent.runners.remote.cli import main
                main()
            
            # Help should exit with 0
            assert exc_info.value.code == 0
            
            # Check help output
            captured = capsys.readouterr()
            assert "Submit a task to the remote agent worker queue" in captured.out
            assert "--description" in captured.out
            assert "--config" in captured.out


class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete CLI workflow."""
        # Create a mock message queue provider
        mock_mq = Mock(spec=MQProvider)
        mock_mq.initialized = True
        mock_mq.publish = AsyncMock()
        
        # Create mock config
        mock_config = Mock()
        mock_config.cli.mq_provider_name = "test_mq"
        mock_config.cli.task_queue = "tasks"
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config',
                  return_value=mock_mq):
            with patch('flowlib.agent.runners.remote.cli.load_remote_config',
                      return_value=mock_config):
                
                # Simulate CLI call
                await submit_task(
                    mq_provider_name="test_mq",
                    queue_name="tasks",
                    task_description="Integration test task"
                )
                
                # Verify the message was published
                assert mock_mq.publish.called
                message = mock_mq.publish.call_args[1]['message']
                assert isinstance(message, AgentTaskMessage)
                assert message.task_description == "Integration test task"